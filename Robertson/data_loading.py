import jax
import jax.numpy as jnp

## Load Data

Train_Data = jnp.log10(jnp.load("data/train_sol_data.npy"))[:, 1:, :]
Train_odeParameters = jnp.load("data/train_rates_data.npy")

Validation_Data = jnp.log10(jnp.load("data/validation_sol_data.npy"))[:, 1:, :]
Validation_odeParameters = jnp.load("data/validation_rates_data.npy")

Test_Data = jnp.log10(jnp.load("data/test_sol_data.npy"))[:, 1:, :]
Test_odeParameters = jnp.load("data/test_rates_data.npy") 

# Solution data organized as (Num Samples, Num Timesteps, Num Dimensions)
# Parameter data organized as (Num Samples, Num Parameters)

print(Train_Data.shape, Train_odeParameters.shape)
print(Validation_Data.shape, Validation_odeParameters.shape)
print(Test_Data.shape, Test_odeParameters.shape)
