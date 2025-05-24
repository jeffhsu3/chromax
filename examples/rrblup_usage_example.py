"""
Example script demonstrating the usage of RRBLUPTraitModel within the Simulator.

This script covers:
1. Setting up parameters and generating dummy data for genotypes and phenotypes.
2. Initializing the Simulator with `gebv_model_type="rrblup"`.
3. Training the RR-BLUP model using the `train_rrblup_model` method.
4. Predicting Genomic Estimated Breeding Values (GEBVs) on a new population.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import pandas as pd

from chromax.simulator import Simulator
# Assuming Population is a type hint, actual data will be JAX arrays.
# from chromax.typing import Population
# For running standalone, genetic_map and genome are usually loaded from chromax.sample_data
# Here, we'll create minimal versions if sample_data is not directly usable or for simplicity.

def run_rrblup_example():
    """Runs the RR-BLUP example."""
    print("Starting RR-BLUP usage example...\n")

    # --- 1. Setup Parameters and Data ---
    key = random.key(42)
    key_gm, key_train_pop, key_train_pheno, key_pred_pop = random.split(key, 4)

    n_individuals_train = 100
    n_individuals_predict = 50
    n_markers = 500  # Number of markers
    n_traits = 1     # Number of traits

    trait_names = ["Yield"] # Trait name for the genetic map and simulator

    # Create a minimal genetic_map DataFrame
    # In a real scenario, this would be loaded from chromax.sample_data.genetic_map or a file
    genetic_map_df = pd.DataFrame({
        'CHR.PHYS': [1] * n_markers, # Dummy chromosome
        'cM': jnp.linspace(0, 100, n_markers), # Dummy positions
        # Marker effects for the 'Yield' trait (used by linear model, but column needed)
        trait_names[0]: random.normal(key_gm, (n_markers,))
    })
    print(f"Created genetic map with {n_markers} markers for trait '{trait_names[0]}'.\n")

    # Create a training population (genotypes)
    # Shape: (n_individuals, n_markers, ploidy=2)
    training_population = random.randint(key_train_pop,
                                         (n_individuals_train, n_markers, 2),
                                         0, 2, dtype=jnp.int8)
    print(f"Generated training population with shape: {training_population.shape}")

    # Create training phenotypes
    # Shape: (n_individuals, n_traits)
    # Making phenotypes somewhat related to summed genotypes for a bit of realism
    pheno_genetic_effect = training_population.sum(axis=(1, 2)) * 0.05 # Sum across markers and ploidy
    # Ensure pheno_genetic_effect is (n_individuals_train, 1) for broadcasting with noise
    pheno_genetic_effect = pheno_genetic_effect.reshape(-1, 1)
    
    noise = random.normal(key_train_pheno, (n_individuals_train, n_traits)) * 0.5
    training_phenotypes = pheno_genetic_effect + 10 + noise # Add a base value and noise
    print(f"Generated training phenotypes with shape: {training_phenotypes.shape}\n")

    # Create a prediction population (genotypes)
    prediction_population = random.randint(key_pred_pop,
                                           (n_individuals_predict, n_markers, 2),
                                           0, 2, dtype=jnp.int8)
    print(f"Generated prediction population with shape: {prediction_population.shape}\n")


    # --- 2. Simulator Initialization for RR-BLUP ---
    # Initialize the Simulator, specifying "rrblup" as the GEBV model type
    # and providing a default lambda value.
    lambda_initial = 0.5
    simulator = Simulator(
        genetic_map=genetic_map_df,
        trait_names=trait_names,
        gebv_model_type="rrblup",
        lambda_rrblup=lambda_initial, # This can be overridden in train_rrblup_model
        seed=42 # Seed for simulator's internal randomness (e.g. for GxE model if used)
    )
    print(f"Simulator initialized with gebv_model_type='rrblup' and lambda_rrblup={lambda_initial}.\n")

    # --- 3. Model Training ---
    # Train the RR-BLUP model using the training population and phenotypes.
    # We can use the lambda_rrblup from initialization or specify a new one.
    # Let's specify a new one for demonstration.
    lambda_for_training = 0.75
    print(f"Starting RR-BLUP model training with lambda_val={lambda_for_training}...")
    simulator.train_rrblup_model(
        training_population=training_population,
        training_phenotypes=training_phenotypes,
        lambda_val=lambda_for_training # Overrides lambda_rrblup from __init__
    )
    print("RR-BLUP model training complete.\n")

    # --- 4. GEBV Prediction ---
    # Predict GEBVs on the new prediction_population.
    # The GEBV method will use the trained RR-BLUP model.
    print("Predicting GEBVs for the prediction population...")
    predicted_gebvs_df = simulator.GEBV(prediction_population)

    print("\nPredicted GEBVs (first 5 individuals):")
    print(predicted_gebvs_df.head())
    print(f"\nShape of predicted GEBVs DataFrame: {predicted_gebvs_df.shape}")

    # --- Example of GEBV prediction without training (should raise error) ---
    print("\n--- Example of GEBV prediction without training (expecting error) ---")
    simulator_not_trained = Simulator(
        genetic_map=genetic_map_df,
        trait_names=trait_names,
        gebv_model_type="rrblup",
        lambda_rrblup=0.1,
        seed=123
    )
    try:
        print("Attempting GEBV prediction with an untrained RR-BLUP model...")
        gebvs_error = simulator_not_trained.GEBV(prediction_population)
        print(f"GEBVs (should not be reached): {gebvs_error.head()}")
    except RuntimeError as e:
        print(f"Successfully caught expected error: {e}")
    except Exception as e:
        print(f"Caught an unexpected error: {e}")

    print("\nRR-BLUP usage example finished.")

if __name__ == '__main__':
    run_rrblup_example()
