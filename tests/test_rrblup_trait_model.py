import unittest
import jax
import jax.numpy as jnp
import jax.random as random

from chromax.rrblup_trait_model import RRBLUPTraitModel, _check_inputs
from chromax.typing import Population # Assuming Population is Array, or define a placeholder if needed for tests

# If chromax.typing.Population is not a simple JAX array type, we might need a placeholder for tests
# For now, we assume it's compatible with JAX arrays.

class TestRRBLUPTraitModel(unittest.TestCase):
    def setUp(self):
        self.key = random.key(42)

        # Small data for manual calculation verification
        self.n_individuals_small = 4
        self.n_markers_small = 3
        self.n_traits_small_st = 1 # Single trait
        self.lambda_val = 0.1

        key_pop_small, key_pheno_small_st, self.key = random.split(self.key, 3)
        # Population: (n_individuals, n_markers, ploidy=2)
        self.population_small = random.randint(key_pop_small,
                                               (self.n_individuals_small, self.n_markers_small, 2),
                                               0, 2, dtype=jnp.int8)
        # Phenotypes: (n_individuals, n_traits)
        self.phenotypes_small_st = random.normal(key_pheno_small_st,
                                                 (self.n_individuals_small, self.n_traits_small_st)) * 2 + 5

        # Larger data for general testing
        self.n_individuals_large = 50
        self.n_markers_large = 100
        self.n_traits_large_mt = 3 # Multiple traits

        key_pop_large, key_pheno_large_mt, self.key = random.split(self.key, 3)
        self.population_large = random.randint(key_pop_large,
                                               (self.n_individuals_large, self.n_markers_large, 2),
                                               0, 2, dtype=jnp.int8)
        self.phenotypes_large_mt = random.normal(key_pheno_large_mt,
                                                  (self.n_individuals_large, self.n_traits_large_mt)) * 2 + 5
        
        # Phenotypes for multi-trait small case
        key_pheno_small_mt, self.key = random.split(self.key)
        self.n_traits_small_mt = 2
        self.phenotypes_small_mt = random.normal(key_pheno_small_mt,
                                                 (self.n_individuals_small, self.n_traits_small_mt)) * 2 + 5


    def test_initialization(self):
        model = RRBLUPTraitModel(n_markers=self.n_markers_large, n_traits=self.n_traits_large_mt)
        self.assertIsNone(model.marker_effects)
        self.assertIsNone(model.intercept)
        self.assertFalse(model.is_trained)
        self.assertEqual(model.n_markers, self.n_markers_large)
        self.assertEqual(model.n_traits, self.n_traits_large_mt)

    def test_train_and_predict_single_trait(self):
        model = RRBLUPTraitModel(n_markers=self.n_markers_small, n_traits=self.n_traits_small_st)
        model.train(self.population_small, self.phenotypes_small_st, self.lambda_val)

        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.marker_effects)
        self.assertIsNotNone(model.intercept)
        self.assertEqual(model.marker_effects.shape, (self.n_markers_small, self.n_traits_small_st))
        self.assertEqual(model.intercept.shape, (self.n_traits_small_st,))

        # Prediction
        gebvs = model(self.population_small)
        self.assertEqual(gebvs.shape, (self.n_individuals_small, self.n_traits_small_st))

        # Manual calculation verification
        Z = self.population_small.sum(axis=-1, dtype=jnp.float32)
        Z_centered = Z - Z.mean(axis=0)
        
        intercept_expected = self.phenotypes_small_st.mean(axis=0)
        y_centered = self.phenotypes_small_st - intercept_expected
        
        # u_hat = (Z_centered' @ Z_centered + lambda_val * I)^-1 @ Z_centered' @ y_centered
        A = Z_centered.T @ Z_centered + self.lambda_val * jnp.eye(self.n_markers_small)
        u_hat_expected = jnp.linalg.solve(A, Z_centered.T @ y_centered)
        
        self.assertTrue(jnp.allclose(model.intercept, intercept_expected))
        self.assertTrue(jnp.allclose(model.marker_effects, u_hat_expected.reshape(self.n_markers_small, 1)))

        # GEBVs: Z_pred @ marker_effects + intercept
        # Here Z_pred is the same Z used for training
        gebvs_expected = (Z @ model.marker_effects) + model.intercept
        self.assertTrue(jnp.allclose(gebvs, gebvs_expected))

    def test_train_and_predict_multiple_traits(self):
        model = RRBLUPTraitModel(n_markers=self.n_markers_small, n_traits=self.n_traits_small_mt) # Use small data for faster test
        model.train(self.population_small, self.phenotypes_small_mt, self.lambda_val)

        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.marker_effects)
        self.assertIsNotNone(model.intercept)
        self.assertEqual(model.marker_effects.shape, (self.n_markers_small, self.n_traits_small_mt))
        self.assertEqual(model.intercept.shape, (self.n_traits_small_mt,))

        gebvs = model(self.population_small)
        self.assertEqual(gebvs.shape, (self.n_individuals_small, self.n_traits_small_mt))

        # Manual calculation verification for multiple traits
        Z = self.population_small.sum(axis=-1, dtype=jnp.float32)
        Z_centered = Z - Z.mean(axis=0)
        
        intercept_expected = self.phenotypes_small_mt.mean(axis=0)
        y_centered = self.phenotypes_small_mt - intercept_expected # (n_individuals, n_traits)
        
        A = Z_centered.T @ Z_centered + self.lambda_val * jnp.eye(self.n_markers_small)
        # Z_centered.T @ y_centered gives (n_markers, n_traits)
        u_hat_expected = jnp.linalg.solve(A, Z_centered.T @ y_centered) # solve handles multiple RHS
        
        self.assertTrue(jnp.allclose(model.intercept, intercept_expected))
        self.assertTrue(jnp.allclose(model.marker_effects, u_hat_expected))

        gebvs_expected = (Z @ model.marker_effects) + model.intercept
        self.assertTrue(jnp.allclose(gebvs, gebvs_expected))


    def test_predict_before_train(self):
        model = RRBLUPTraitModel(n_markers=self.n_markers_large, n_traits=self.n_traits_large_mt)
        with self.assertRaisesRegex(Exception, "Model not trained yet"):
            model(self.population_large)

    def test_input_validation_mismatched_individuals(self):
        model = RRBLUPTraitModel(n_markers=self.n_markers_small, n_traits=self.n_traits_small_st)
        phenotypes_wrong_individuals = random.normal(self.key, (self.n_individuals_small + 1, self.n_traits_small_st))
        
        with self.assertRaisesRegex(AssertionError, "Number of individuals in population"):
            model.train(self.population_small, phenotypes_wrong_individuals, self.lambda_val)

    def test_input_validation_mismatched_markers(self):
        model = RRBLUPTraitModel(n_markers=self.n_markers_small + 1, n_traits=self.n_traits_small_st) # Model expects more markers
        
        with self.assertRaisesRegex(AssertionError, "Number of markers in population"):
            model.train(self.population_small, self.phenotypes_small_st, self.lambda_val)

    def test_check_inputs_direct_call(self): # Test _check_inputs directly
        # Correct case
        _check_inputs(self.population_small, self.phenotypes_small_st, self.n_markers_small)

        # Incorrect population ndim
        with self.assertRaisesRegex(AssertionError, "Population should have 3 dimensions"):
            _check_inputs(self.population_small.sum(axis=-1), self.phenotypes_small_st, self.n_markers_small)

        # Incorrect phenotypes ndim
        with self.assertRaisesRegex(AssertionError, "Phenotypes should have 2 dimensions"):
            _check_inputs(self.population_small, self.phenotypes_small_st.flatten(), self.n_markers_small)
        
        # Mismatched individuals
        pheno_wrong_ind = self.phenotypes_small_st[:-1]
        with self.assertRaisesRegex(AssertionError, "Number of individuals in population"):
            _check_inputs(self.population_small, pheno_wrong_ind, self.n_markers_small)

        # Mismatched markers
        with self.assertRaisesRegex(AssertionError, "Number of markers in population"):
            _check_inputs(self.population_small, self.phenotypes_small_st, self.n_markers_small -1)


if __name__ == '__main__':
    unittest.main()
