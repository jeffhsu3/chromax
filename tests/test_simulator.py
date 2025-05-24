import warnings

import jax
import numpy as np
import pandas as pd
import pytest

from chromax import Simulator
from chromax.sample_data import genetic_map, genome
from mock_simulator import MockSimulator


@pytest.mark.parametrize("idx", [0, 1])
def test_cross_r(idx):
    n_markers = 1000
    recombination_vec = np.zeros(n_markers, dtype="bool")
    recombination_vec[0] = idx
    simulator = MockSimulator(recombination_vec=recombination_vec)

    ploidy = 4
    size = (1, 2, simulator.n_markers, ploidy)
    parents = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])

    new_pop = simulator.cross(parents)

    assert new_pop.shape == (1, simulator.n_markers, ploidy)

    ind = new_pop[0]
    for i in range(ploidy):
        pair_chr_idx = i % 2
        assert np.all(ind[:, i] == parents[0, pair_chr_idx, :, i - pair_chr_idx + idx])


def test_equal_parents():
    simulator = Simulator(genetic_map=genetic_map)

    ploidy = 4
    parents = np.zeros((1, 2, simulator.n_markers, ploidy), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 0)

    parents = np.ones((1, 2, simulator.n_markers, ploidy), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 1)


def test_ad_hoc_cross():
    rec_vec = np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8
    )

    simulator = MockSimulator(recombination_vec=rec_vec)
    population = simulator.load_population(2, ploidy=4)
    parents = population[np.array([[0, 1]])]
    child = simulator.cross(parents)

    assert child.shape == (1, *population[0].shape)

    chr_idx = 0
    for mrk_idx, rec_prob in enumerate(rec_vec):
        if rec_prob == 1:
            chr_idx = 1 - chr_idx
        assert child[1, mrk_idx, 0] == population[0, mrk_idx, chr_idx]
        assert child[1, mrk_idx, 1] == population[1, mrk_idx, chr_idx]
        assert child[1, mrk_idx, 2] == population[0, mrk_idx, 2 + chr_idx]
        assert child[1, mrk_idx, 3] == population[1, mrk_idx, 2 + chr_idx]


def test_cross_two_times():
    n_markers = 100_000
    n_ind = 2
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=4)

    parents = population[np.array([[0, 1], [0, 1]])]
    children = simulator.cross(parents)

    assert np.any(children[0] != children[1])


def test_double_haploid():
    n_markers = 1000
    n_ind = 100
    n_offspring = 10
    ploidy = 4

    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    new_pop = simulator.double_haploid(population, n_offspring=n_offspring)
    assert new_pop.shape == (len(population), n_offspring, *population.shape[1:])
    assert np.all(new_pop[..., 0] == new_pop[..., 1])
    assert np.all(new_pop[..., 2] == new_pop[..., 3])

    new_pop = simulator.double_haploid(population)
    assert new_pop.shape == population.shape
    assert np.all(new_pop[..., 0] == new_pop[..., 1])
    assert np.all(new_pop[..., 2] == new_pop[..., 3])


def test_diallel():
    n_markers = 1000
    n_ind = 100
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    diallel_indices = simulator._diallel_indices(np.arange(10))
    assert len(np.unique(diallel_indices, axis=0)) == 45

    new_pop = simulator.diallel(population)
    assert new_pop.shape == (n_ind * (n_ind - 1) // 2, n_markers, ploidy)

    new_pop = simulator.diallel(population, n_offspring=10)
    assert new_pop.shape == (n_ind * (n_ind - 1) // 2, 10, n_markers, ploidy)


def test_select():
    n_markers = 1000
    n_ind = 100
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)
    pop_GEBV = simulator.GEBV(population)

    k = 10
    selected_pop, selected_indices = simulator.select(population, k=k)
    assert selected_pop.shape == (k, n_markers, ploidy)
    assert selected_indices.shape == (k,)
    selected_GEBV = simulator.GEBV(selected_pop)
    assert np.all(selected_GEBV.mean() > pop_GEBV.mean())
    assert np.all(selected_GEBV.max() == pop_GEBV.max())
    assert np.all(selected_GEBV.min() > pop_GEBV.min())
    GEBV_indices = pop_GEBV.iloc[selected_indices]
    assert np.all(GEBV_indices.reset_index(drop=True) == selected_GEBV)

    k = 5
    dh = simulator.double_haploid(population, n_offspring=100)
    selected_dh, selected_indices = simulator.select(dh, k=k)
    assert selected_dh.shape == (n_ind, k, n_markers, ploidy)
    assert selected_indices.shape == (n_ind, k)
    for i in range(n_ind):
        dh_GEBV = simulator.GEBV(dh[i])
        selected_GEBV = simulator.GEBV(selected_dh[i])
        assert np.all(selected_GEBV.mean() > dh_GEBV.mean())
        assert np.all(selected_GEBV.max() == dh_GEBV.max())
        assert np.all(selected_GEBV.min() > dh_GEBV.min())
        GEBV_indices = dh_GEBV.iloc[selected_indices[i]]
        assert np.all(GEBV_indices.reset_index(drop=True) == selected_GEBV)


def test_random_crosses():
    n_markers = 1000
    n_ind = 100
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    n_crosses = 300
    new_pop, _ = simulator.random_crosses(population, n_crosses=n_crosses)
    assert new_pop.shape == (n_crosses, n_markers, ploidy)

    n_offspring = 10
    new_pop, _ = simulator.random_crosses(
        population=population, n_crosses=n_crosses, n_offspring=n_offspring
    )
    assert new_pop.shape == (n_crosses, n_offspring, n_markers, ploidy)


def test_multi_trait():
    trait_names = [
        "Heading Date",
        "Protein Content",
        "Plant Height",
        "Thousand Kernel Weight",
        "Yield",
        "Fusarium Head Blight",
        "Spike Emergence Period",
    ]
    simulator = Simulator(genetic_map=genetic_map, trait_names=trait_names)
    population = simulator.load_population(genome)

    gebv_shape = len(population), len(trait_names)
    assert simulator.GEBV(population).shape == gebv_shape


def test_device():
    local_devices = jax.local_devices()
    if len(local_devices) == 1:
        warnings.warn("Device testing skipped because there is only one device.")
        return

    device = local_devices[1]
    simulator = Simulator(genetic_map=genetic_map, device=device)

    population = simulator.load_population(genome)
    assert population.device_buffer.device() == device

    GEBV = simulator.GEBV_model(population)
    assert GEBV.device_buffer.device() == device

    selected_pop, _ = simulator.select(population, k=10)
    assert selected_pop.device_buffer.device() == device

    diallel = simulator.diallel(selected_pop, n_offspring=10)
    assert diallel.device_buffer.device() == device

    dh_pop = simulator.double_haploid(diallel)
    assert dh_pop.device_buffer.device() == device

    cross_indices = np.array([[1, 5], [3, 10], [100, 2], [7, 93], [28, 41]])
    new_pop = simulator.cross(dh_pop[cross_indices])
    assert new_pop.device_buffer.device() == device


def test_seed_deterministic():
    n_ind = 100
    ploidy = 4
    simulator1 = Simulator(genetic_map=genetic_map, seed=7)
    simulator2 = Simulator(genetic_map=genetic_map, seed=7)
    mock_simulator = MockSimulator(n_markers=simulator1.n_markers)
    population = mock_simulator.load_population(n_ind, ploidy=ploidy)

    new_pop1, _ = simulator1.random_crosses(population, n_crosses=10)
    new_pop2, _ = simulator2.random_crosses(population, n_crosses=10)

    assert np.all(new_pop1 == new_pop2)


def test_gebv():
    n_markers, n_ind = 100, 10
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    gebv_pandas = simulator.GEBV(population)
    assert len(gebv_pandas) == n_ind
    assert isinstance(gebv_pandas, pd.DataFrame)

    gebv_array = simulator.GEBV(population, raw_array=True)
    assert len(gebv_array) == n_ind
    assert np.all(gebv_pandas.values == gebv_array)


def test_phenotyping():
    n_markers, n_ind = 100, 10
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    phenotype = simulator.phenotype(population, num_environments=4)
    assert len(phenotype) == n_ind

    environments = np.random.uniform(-1, 1, size=(8,))
    _ = simulator.phenotype(population, environments=environments)
    assert len(phenotype) == n_ind

    with pytest.raises(ValueError):
        simulator.phenotype(population, num_environments=8, environments=environments)


def test_mutation():
    n_markers, n_ind = 1000, 10
    rec_vec = np.zeros(n_markers)
    simulator = MockSimulator(n_markers=n_markers, recombination_vec=rec_vec)
    population = simulator.load_population(n_ind, ploidy=2)
    parents = np.transpose(population, axes=(0, 2, 1))
    parents = np.broadcast_to(parents[..., None], (n_ind, 2, n_markers, 2))

    assert np.all(simulator.cross(parents) == population)

    simulator = MockSimulator(n_markers, rec_vec, mutation_probability=1)
    assert np.all(simulator.cross(parents) != population)

    simulator = MockSimulator(n_markers, rec_vec, mutation_probability=0.5)
    assert not np.all(simulator.cross(parents) == population)
    assert not np.all(simulator.cross(parents) != population)


# Tests for RRBLUPTraitModel integration with Simulator
class TestSimulatorRRBLUP:
    @pytest.fixture
    def key(self):
        return jax.random.key(42)

    @pytest.fixture
    def sample_genetic_map_df(self):
        # Using the sample data provided by chromax
        return genetic_map.copy()

    @pytest.fixture
    def sample_genome_data(self, key):
        # A smaller genome for faster tests
        n_individuals, n_markers, ploidy = 10, 100, 2 # Reduced n_markers from sample_data.genome
        
        # Ensure n_markers in sample_genome_data is consistent with sample_genetic_map_df for RRBLUP
        # The sample_genetic_map has 9839 markers. Let's use a subset of that for testing.
        # For RRBLUP, n_markers in model must match n_markers in population.
        # The genetic_map fixture will determine n_markers for the simulator.
        
        # Let's create a dummy genome that matches a smaller genetic map for RRBLUP tests
        # if we decide to use a smaller map. For now, use full map and derive genome.
        
        # Using the sample genome provided by chromax, but potentially a subset of individuals
        # sample_genome_array = genome[:n_individuals, :n_markers_for_sim, :]
        # For now, let's assume we'll use the full genetic_map, so n_markers will be its length.
        # We need a population that matches this.
        
        # Create a random population for testing if sample_data.genome is too large or not flexible enough
        # This population will be used as the `training_population` or `prediction_population`
        
        # The simulator will load its n_markers from the genetic_map.
        # So, the test population should have the same n_markers.
        # n_markers_from_map = len(self.sample_genetic_map_df()) # This cannot be called here.
        # Instead, the simulator instance will tell us n_markers.
        
        # This fixture will provide a small population. The number of markers
        # should be adjusted in the test methods based on the simulator instance.
        pop_data = jax.random.randint(key, (n_individuals, 100, ploidy), 0, 2, dtype=jnp.int8)
        return pop_data


    def test_simulator_init_rrblup(self, sample_genetic_map_df):
        lambda_val = 0.05
        simulator = Simulator(
            genetic_map=sample_genetic_map_df,
            gebv_model_type="rrblup",
            lambda_rrblup=lambda_val,
            trait_names=["Yield"] # Assuming 'Yield' is a column in sample_genetic_map_df
        )
        assert isinstance(simulator.GEBV_model, RRBLUPTraitModel)
        assert simulator.lambda_rrblup == lambda_val
        assert simulator.GEBV_model.n_markers == len(sample_genetic_map_df)
        assert simulator.GEBV_model.n_traits == 1

    def test_train_rrblup_model_method(self, sample_genetic_map_df, key):
        n_individuals_train = 20
        n_traits = 1 # For simplicity, matching trait_names=["Yield"]
        
        simulator = Simulator(
            genetic_map=sample_genetic_map_df,
            gebv_model_type="rrblup",
            lambda_rrblup=0.1,
            trait_names=["Yield"],
            seed=42
        )
        n_markers_sim = simulator.n_markers # Get n_markers from simulator instance

        # Create training population and phenotypes matching simulator's n_markers
        training_population = jax.random.randint(key, (n_individuals_train, n_markers_sim, 2), 0, 2, dtype=jnp.int8)
        training_phenotypes = jax.random.normal(key, (n_individuals_train, n_traits)) * 3 + 10
        
        simulator.train_rrblup_model(training_population, training_phenotypes)
        assert simulator.GEBV_model.is_trained
        assert simulator.GEBV_model.marker_effects is not None
        assert simulator.GEBV_model.intercept is not None
        assert simulator.GEBV_model.marker_effects.shape == (n_markers_sim, n_traits)
        assert simulator.GEBV_model.intercept.shape == (n_traits,)

    def test_train_rrblup_model_no_lambda(self, sample_genetic_map_df, key):
        n_individuals_train = 10
        n_traits = 1
        simulator = Simulator(
            genetic_map=sample_genetic_map_df,
            gebv_model_type="rrblup",
            # lambda_rrblup is deliberately not set here
            trait_names=["Yield"],
            seed=42
        )
        n_markers_sim = simulator.n_markers
        training_population = jax.random.randint(key, (n_individuals_train, n_markers_sim, 2), 0, 2, dtype=jnp.int8)
        training_phenotypes = jax.random.normal(key, (n_individuals_train, n_traits))

        with pytest.raises(ValueError, match="lambda_val must be provided"):
            simulator.train_rrblup_model(training_population, training_phenotypes) # No lambda_val argument

    def test_train_rrblup_on_linear_model(self, sample_genetic_map_df, key):
        n_individuals_train = 10
        n_traits = 1
        simulator = Simulator(
            genetic_map=sample_genetic_map_df,
            gebv_model_type="linear", # Default or explicitly linear
            trait_names=["Yield"],
            seed=42
        )
        n_markers_sim = simulator.n_markers
        training_population = jax.random.randint(key, (n_individuals_train, n_markers_sim, 2), 0, 2, dtype=jnp.int8)
        training_phenotypes = jax.random.normal(key, (n_individuals_train, n_traits))

        with pytest.raises(TypeError, match="Current GEBV model is not trainable with RR-BLUP"):
            simulator.train_rrblup_model(training_population, training_phenotypes)

    def test_gebv_with_rrblup_trained(self, sample_genetic_map_df, key):
        n_individuals_train = 30
        n_individuals_pred = 5
        n_traits = 1 
        lambda_val = 0.01

        simulator = Simulator(
            genetic_map=sample_genetic_map_df,
            gebv_model_type="rrblup",
            lambda_rrblup=lambda_val,
            trait_names=["Yield"],
            seed=42
        )
        n_markers_sim = simulator.n_markers
        
        key_train_pop, key_train_pheno, key_pred_pop = jax.random.split(key, 3)

        training_population = jax.random.randint(key_train_pop, (n_individuals_train, n_markers_sim, 2), 0, 2, dtype=jnp.int8)
        training_phenotypes = jax.random.normal(key_train_pheno, (n_individuals_train, n_traits)) * 4 + 12
        
        simulator.train_rrblup_model(training_population, training_phenotypes) # Uses lambda_rrblup from init

        prediction_population = jax.random.randint(key_pred_pop, (n_individuals_pred, n_markers_sim, 2), 0, 2, dtype=jnp.int8)
        
        gebvs_df = simulator.GEBV(prediction_population)
        
        assert isinstance(gebvs_df, pd.DataFrame)
        assert gebvs_df.shape == (n_individuals_pred, n_traits)
        assert not gebvs_df.isnull().values.any() # Check for NaNs
        
        # Optional: Check if values are somewhat reasonable (e.g. not all zero if inputs are not all zero)
        # This is a weak check, strong check would be manual calculation.
        assert not np.allclose(gebvs_df.values, 0)

        # Manual calculation for one individual (first one) as a sanity check
        Z_pred_one = prediction_population[0:1].sum(axis=-1, dtype=jnp.float32) # Shape (1, n_markers_sim)
        
        # The RRBLUPTraitModel's __call__ is Z_pred @ marker_effects + intercept
        # where marker_effects were derived from Z_centered_train
        # and intercept is phenotypes_train.mean(axis=0)
        
        # To correctly verify, we must follow the exact logic of RRBLUPTraitModel:
        # Z_train = training_population.sum(axis=-1, dtype=jnp.float32)
        # Z_train_centered = Z_train - Z_train.mean(axis=0)
        # intercept_expected = training_phenotypes.mean(axis=0)
        # y_centered = training_phenotypes - intercept_expected
        # A = Z_train_centered.T @ Z_train_centered + lambda_val * np.eye(n_markers_sim)
        # u_hat_expected = np.linalg.solve(A, Z_train_centered.T @ y_centered)

        # The model stores u_hat_expected as self.marker_effects and intercept_expected as self.intercept
        # So, gebv_manual = Z_pred_one @ simulator.GEBV_model.marker_effects + simulator.GEBV_model.intercept
        
        gebv_manual_first_ind = Z_pred_one @ simulator.GEBV_model.marker_effects + simulator.GEBV_model.intercept
        
        assert jnp.allclose(gebvs_df.iloc[0].values, gebv_manual_first_ind.flatten(), atol=1e-5)


    def test_gebv_with_rrblup_not_trained(self, sample_genetic_map_df, key):
        n_individuals_pred = 5
        simulator = Simulator(
            genetic_map=sample_genetic_map_df,
            gebv_model_type="rrblup",
            lambda_rrblup=0.1,
            trait_names=["Yield"],
            seed=42
        )
        n_markers_sim = simulator.n_markers
        prediction_population = jax.random.randint(key, (n_individuals_pred, n_markers_sim, 2), 0, 2, dtype=jnp.int8)

        with pytest.raises(Exception, match="Model not trained yet"):
            simulator.GEBV(prediction_population)

    def test_init_rrblup_multiple_traits(self, sample_genetic_map_df):
        trait_names_multi = ["Yield", "Plant Height"] # Assuming these are in sample_genetic_map_df
        # Ensure sample_genetic_map_df actually has these columns
        # For the purpose of this test, let's assume genetic_map fixture is modified or these columns exist
        # If not, this test might fail at Simulator init if it tries to access non-existent columns for linear model part
        # However, for RRBLUP, it only needs n_traits, not the actual effect columns from map at init.
        
        # To make this test robust, we can create a dummy genetic map for it
        dummy_map_data = {
            "CHR.PHYS": [1]*100,
            "cM": np.linspace(0,10,100),
            "Yield": np.random.rand(100),
            "Plant Height": np.random.rand(100)
        }
        dummy_genetic_map = pd.DataFrame(dummy_map_data)


        simulator = Simulator(
            genetic_map=dummy_genetic_map, # Use a map known to have these traits
            gebv_model_type="rrblup",
            lambda_rrblup=0.1,
            trait_names=trait_names_multi,
            seed=42
        )
        assert isinstance(simulator.GEBV_model, RRBLUPTraitModel)
        assert simulator.GEBV_model.n_markers == 100
        assert simulator.GEBV_model.n_traits == len(trait_names_multi)

    def test_train_rrblup_model_multiple_traits(self, key):
        n_individuals_train = 20
        n_markers_sim = 50 # Using a smaller, independent n_markers for this test
        n_traits_multi = 2
        
        # Create a dummy genetic map that fits these dimensions for the simulator
        dummy_map_data = {
            "CHR.PHYS": [1]*n_markers_sim,
            "cM": np.linspace(0,10,n_markers_sim),
            # Dummy effect columns, names matching trait_names_multi
            "Trait1": np.random.rand(n_markers_sim), 
            "Trait2": np.random.rand(n_markers_sim)
        }
        dummy_genetic_map = pd.DataFrame(dummy_map_data)
        trait_names_multi = ["Trait1", "Trait2"]

        simulator = Simulator(
            genetic_map=dummy_genetic_map,
            gebv_model_type="rrblup",
            lambda_rrblup=0.1,
            trait_names=trait_names_multi, # Pass the trait names
            seed=42
        )
        
        assert simulator.n_markers == n_markers_sim
        assert len(simulator.trait_names) == n_traits_multi
        assert simulator.GEBV_model.n_traits == n_traits_multi


        training_population = jax.random.randint(key, (n_individuals_train, n_markers_sim, 2), 0, 2, dtype=jnp.int8)
        training_phenotypes = jax.random.normal(key, (n_individuals_train, n_traits_multi)) * 3 + 10
        
        simulator.train_rrblup_model(training_population, training_phenotypes)
        assert simulator.GEBV_model.is_trained
        assert simulator.GEBV_model.marker_effects.shape == (n_markers_sim, n_traits_multi)
        assert simulator.GEBV_model.intercept.shape == (n_traits_multi,)

        # Test GEBV prediction
        n_individuals_pred = 5
        prediction_population = jax.random.randint(key, (n_individuals_pred, n_markers_sim, 2), 0, 2, dtype=jnp.int8)
        gebvs_df = simulator.GEBV(prediction_population)
        assert gebvs_df.shape == (n_individuals_pred, n_traits_multi)
        assert list(gebvs_df.columns) == trait_names_multi

# Need to import RRBLUPTraitModel if not already at the top of the file
# from chromax.rrblup_trait_model import RRBLUPTraitModel
# Assuming chromax.sample_data.genetic_map and genome are pandas DataFrames / numpy arrays
# For RRBLUP, make sure that population data used for training/prediction has n_markers
# that matches n_markers in the genetic_map used to initialize Simulator (for RRBLUP model's n_markers).
