""" Module for the data likelihood/emissions of the HMM. """

import jax.numpy as jnp
import jax.scipy.stats as jss

def likelihood(data: jnp.ndarray, states_values:jnp.ndarray)->jnp.ndarray:
    return jss.norm.pdf(data.reshape(-1, 1), loc=0, scale=states_values)