"""
Module for Ridge Regression Best Linear Unbiased Prediction (RR-BLUP) trait model.

This module provides the `RRBLUPTraitModel` class, which implements a trait model
based on RR-BLUP for predicting Genomic Estimated Breeding Values (GEBVs), and
a helper function `_check_inputs` for validating input data dimensions.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from .typing import Population


# Helper function (outside class, but in the same file)
def _check_inputs(
    population: Population,
    phenotypes: Float[Array, "n_individuals n_traits"],
    n_markers_model: int
):
    """
    Checks the validity of inputs for RRBLUPTraitModel's train method.

    Ensures that the population and phenotype data have compatible shapes and
    dimensions with the model's configuration.

    Args:
        population: Population object with genotype data, expected to have shape
            (n_individuals, n_markers_pop, ploidy_sum_axis).
        phenotypes: Phenotype array, expected to have shape (n_individuals, n_traits).
        n_markers_model: The number of markers the RRBLUP model is configured for.

    Raises:
        AssertionError: If any of the input validation checks fail.
    """
    assert population.ndim == 3, (
        f"Population should have 3 dimensions (n_individuals, n_markers, ploidy_sum_axis), "
        f"got {population.ndim}"
    )
    assert phenotypes.ndim == 2, (
        f"Phenotypes should have 2 dimensions (n_individuals, n_traits), "
        f"got {phenotypes.ndim}"
    )
    assert population.shape[0] == phenotypes.shape[0], (
        f"Number of individuals in population ({population.shape[0]}) must match "
        f"number of individuals in phenotypes ({phenotypes.shape[0]})"
    )
    assert population.shape[1] == n_markers_model, (
        f"Number of markers in population ({population.shape[1]}) must match "
        f"number of markers in model ({n_markers_model})"
    )


class RRBLUPTraitModel:
    """
    A trait model using Ridge Regression Best Linear Unbiased Prediction (RR-BLUP).

    This model estimates marker effects and an intercept from a training population
    with known genotypes and phenotypes. It can then be used to predict
    Genomic Estimated Breeding Values (GEBVs) for new populations.

    The core of the model involves solving the mixed model equations for RR-BLUP:
    u_hat = (Z_centered' @ Z_centered + lambda * I)^-1 @ Z_centered' @ y_centered
    where Z_centered is the centered genotype matrix, y_centered are the centered
    phenotypes, lambda is the regularization parameter, and I is the identity matrix.

    Attributes:
        marker_effects (Optional[Float[Array, "n_markers n_traits"]]): Estimated marker
            effects after training. None if not trained.
        intercept (Optional[Float[Array, "n_traits"]]): Estimated intercept(s) for
            each trait after training. None if not trained.
        n_markers (int): The number of markers the model is configured for.
        n_traits (int): The number of traits the model is configured for.
        device (Optional[jax.Device]): The JAX device used for computations.
            If None, JAX's default device is used.
        is_trained (bool): True if the model has been trained, False otherwise.
    """
    marker_effects: Float[Array, "n_markers n_traits"] | None  # pytype: disable=invalid-annotation
    intercept: Float[Array, "n_traits"] | None # pytype: disable=invalid-annotation
    n_markers: int
    n_traits: int
    device: jax.Device | None # pytype: disable=invalid-annotation
    is_trained: bool

    def __init__(self, n_markers: int, n_traits: int, device: jax.Device | None = None):
        """
        Initializes the RRBLUPTraitModel.

        Args:
            n_markers: The number of markers the model will be trained on and
                used for predictions. This should match the number of markers in
                the input population data.
            n_traits: The number of traits the model will predict. This should
                match the second dimension of the input phenotype data.
            device (Optional[jax.Device]): The JAX device (e.g., CPU, GPU, TPU)
                to use for computations. If None, JAX's default device is used.
        """
        self.marker_effects = None
        self.intercept = None
        self.n_markers = n_markers
        self.n_traits = n_traits
        self.device = device  # JAX handles device placement implicitly or via jax.default_device
        self.is_trained = False

    def train(self, population: Population, phenotypes: Float[Array, "n_individuals n_traits"], lambda_val: float):
        """
        Trains the RR-BLUP model.

        Args:
            population (Population): Genotype data for the training population.
                Expected shape is (n_individuals, n_markers, ploidy_sum_axis),
                where `n_markers` must match `self.n_markers`. The ploidy axis
                will be summed to create the genotype matrix Z.
            phenotypes (Float[Array, "n_individuals n_traits"]): Phenotype data
                for the training population. `n_individuals` must match the
                population, and `n_traits` must match `self.n_traits`.
            lambda_val (float): The regularization parameter (lambda) for ridge
                regression. This value controls the shrinkage of marker effects.

        Sets:
            self.marker_effects (Float[Array, "n_markers n_traits"]): The estimated
                marker effects.
            self.intercept (Float[Array, "n_traits"]): The estimated intercept for
                each trait.
            self.is_trained (bool): Set to True after successful training.
        """
        _check_inputs(population, phenotypes, self.n_markers)

        # Genotype Matrix (Z)
        # Summing ploidy: (n_individuals, n_markers_pop, ploidy_sum_axis) -> (n_individuals, n_markers_pop)
        Z = population.sum(axis=-1, dtype=jnp.float32) # Ensure float for mean calculation
        assert Z.shape[1] == self.n_markers, \
            f"Derived genotype matrix Z has {Z.shape[1]} markers, but model expects {self.n_markers}."

        # Center Z by subtracting its column means
        Z_centered = Z - Z.mean(axis=0)

        # Intercept (mean of phenotypes)
        self.intercept = phenotypes.mean(axis=0)
        assert self.intercept.shape == (self.n_traits,), \
            f"Intercept shape mismatch: expected ({self.n_traits},), got {self.intercept.shape}"

        # Centered Phenotypes (y_centered)
        y_centered = phenotypes - self.intercept # Broadcasting self.intercept (n_traits,) to (n_individuals, n_traits)

        # RR-BLUP Calculation
        # u_hat = (Z_centered' @ Z_centered + lambda_val * I)^-1 @ Z_centered' @ y_centered_trait
        
        ZtZ = Z_centered.T @ Z_centered
        
        # Add lambda * I to ZtZ
        # jnp.eye requires int for shape, ensure self.n_markers is int
        regularization_term = lambda_val * jnp.eye(int(self.n_markers))
        A = ZtZ + regularization_term

        if self.n_traits == 1:
            Zty_trait = Z_centered.T @ y_centered
            u_hat_trait = jax.numpy.linalg.solve(A, Zty_trait)
            self.marker_effects = u_hat_trait.reshape(self.n_markers, 1)
        else:
            # Perform calculation for each trait
            # We can vmap the solve operation over the traits in y_centered
            # y_centered has shape (n_individuals, n_traits)
            # Z_centered.T has shape (n_markers, n_individuals)
            # Z_centered.T @ y_centered will result in (n_markers, n_traits)
            
            Zty = Z_centered.T @ y_centered # Shape: (n_markers, n_traits)

            # We want to solve A @ u_hat_trait = Zty_trait for each trait.
            # A is (n_markers, n_markers)
            # Zty_trait is (n_markers,)
            # u_hat_trait is (n_markers,)
            # We can use jax.lax.map or a simple loop, but vmap is cleaner for linalg.solve
            # jax.numpy.linalg.solve can take a stack of matrices and a stack of vectors.
            # So if Zty is (n_markers, n_traits), it will solve for each column.
            
            u_hat = jax.numpy.linalg.solve(A, Zty) # u_hat will have shape (n_markers, n_traits)
            self.marker_effects = u_hat

        assert self.marker_effects is not None
        assert self.marker_effects.shape == (self.n_markers, self.n_traits), \
            f"Marker effects shape mismatch: expected ({self.n_markers}, {self.n_traits}), got {self.marker_effects.shape}"
        
        self.is_trained = True

    def __call__(self, population: Population) -> Float[Array, "n_individuals n_traits"]:
        """
        Predicts Genomic Estimated Breeding Values (GEBVs) for a given population.

        The model must be trained using the `train()` method before predictions
        can be made.

        Args:
            population (Population): Genotype data for the population for which
                GEBVs are to be predicted. Expected shape is
                (n_individuals, n_markers, ploidy_sum_axis), where `n_markers`
                must match `self.n_markers`.

        Returns:
            Float[Array, "n_individuals n_traits"]: Predicted GEBVs for each
            individual and each trait.

        Raises:
            RuntimeError: If the model has not been trained before calling this method.
        """
        if not self.is_trained or self.marker_effects is None or self.intercept is None:
            raise RuntimeError("Model not trained yet. Call train() before making predictions.")

        # Derive the genotype matrix Z_pred from the input population
        # (n_individuals, n_markers, ploidy_sum_axis) -> (n_individuals, n_markers)
        Z_pred = population.sum(axis=-1, dtype=jnp.float32) 
        assert Z_pred.shape[1] == self.n_markers, \
            f"Prediction genotype matrix Z_pred has {Z_pred.shape[1]} markers, model was trained with {self.n_markers}."

        # Calculate GEBVs: gebvs = (Z_pred @ self.marker_effects) + self.intercept
        gebvs = (Z_pred @ self.marker_effects) + self.intercept
        
        assert gebvs.shape[1] == self.n_traits, \
            f"Output GEBVs have {gebvs.shape[1]} traits, but model was trained for {self.n_traits}."
        return gebvs

# Example Usage (commented out, for illustration)
# if __name__ == '__main__':
#     key = jax.random.PRNGKey(0)
#     n_individuals = 100
#     n_markers = 500
#     n_traits = 2
#     ploidy = 2

#     # Dummy Population data (replace with actual Population object if available)
#     # Population shape: (n_individuals, n_markers, ploidy)
#     dummy_population_data = jax.random.randint(key, (n_individuals, n_markers, ploidy), 0, 2)
    
#     # Dummy Phenotype data
#     # Phenotypes shape: (n_individuals, n_traits)
#     key, subkey = jax.random.split(key)
#     dummy_phenotypes = jax.random.normal(subkey, (n_individuals, n_traits)) * 5 + 10

#     # Create a Population-like object (assuming a simple structure for now)
#     class DummyPopulation:
#         def __init__(self, data):
#             self._data = data
#         def sum(self, axis, dtype): # Mocking the sum method used
#             return self._data.sum(axis=axis, dtype=dtype)
#         @property
#         def ndim(self):
#             return self._data.ndim
#         @property
#         def shape(self):
#             return self._data.shape

#     current_population = DummyPopulation(dummy_population_data)

#     # Initialize model
#     model = RRBLUPTraitModel(n_markers=n_markers, n_traits=n_traits)

#     # Train model
#     lambda_val = 0.1
#     model.train(current_population, dummy_phenotypes, lambda_val)
#     print("Model trained.")
#     print("Intercept:", model.intercept)
#     print("Marker effects shape:", model.marker_effects.shape)

#     # Predict
#     # Dummy prediction population (can be the same or different)
#     key, subkey = jax.random.split(key)
#     dummy_pred_population_data = jax.random.randint(subkey, (n_individuals // 2, n_markers, ploidy), 0, 2)
#     pred_population = DummyPopulation(dummy_pred_population_data)
    
#     gebvs = model(pred_population)
#     print("Predicted GEBVs shape:", gebvs.shape)
#     print("First 5 GEBVs:", gebvs[:5, :])

#     # Test with n_traits = 1
#     n_traits_single = 1
#     key, subkey = jax.random.split(key)
#     dummy_phenotypes_single = jax.random.normal(subkey, (n_individuals, n_traits_single)) * 5 + 10
#     model_single = RRBLUPTraitModel(n_markers=n_markers, n_traits=n_traits_single)
#     model_single.train(current_population, dummy_phenotypes_single, lambda_val)
#     print("\nModel trained (single trait).")
#     print("Intercept (single trait):", model_single.intercept)
#     print("Marker effects shape (single trait):", model_single.marker_effects.shape)
#     gebvs_single = model_single(pred_population)
#     print("Predicted GEBVs shape (single trait):", gebvs_single.shape)
#     print("First 5 GEBVs (single trait):", gebvs_single[:5, :])

#     # Test _check_inputs
#     print("\nTesting _check_inputs:")
#     try:
#         _check_inputs(pred_population, dummy_phenotypes_single, n_markers + 1) # Wrong n_markers
#     except AssertionError as e:
#         print(f"Caught expected error: {e}")

#     try:
#         dummy_phenotypes_wrong_ind = jax.random.normal(key, (n_individuals + 10, n_traits_single))
#         _check_inputs(current_population, dummy_phenotypes_wrong_ind, n_markers) # Wrong n_individuals
#     except AssertionError as e:
#         print(f"Caught expected error: {e}")
#     print("Finished example usage.")
# Define Population for type hint if not available (e.g. running this file standalone)
# This is a placeholder. In the actual chromax library, this would be imported.
# from typing import NewType
# Population = NewType('Population', jax.Array)
