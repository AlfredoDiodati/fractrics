"""Statistics to test performances of models"""

import jax.numpy as jnp

def AIC(log_likelihood:float, n_params:int) -> float:
    return 2 * n_params - 2 * log_likelihood

def BIC(log_likelihood:float, n_params:int, n_observations:int) -> float:
    return n_params * jnp.log(n_observations) - 2 * log_likelihood