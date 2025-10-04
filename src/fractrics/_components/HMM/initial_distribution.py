""" Module for the initial distribution of the HMM. """

import jax.numpy as jnp

def check_marg_prob_mass(marg_prob_mass: jnp.ndarray)->None:
    if not jnp.isclose(jnp.sum(marg_prob_mass), 1.0):
        raise ValueError("The marginal probability mass needs to sum to 1.")

# TODO: add generalized discrete distribution function with dependence function.
def factor_pmas(marg_prob_mass: jnp.ndarray, num_latent: int) -> jnp.ndarray:
    grids = jnp.meshgrid(*([marg_prob_mass] * num_latent), indexing="ij")
    stacked = jnp.stack(grids, axis=-1)
    flattened = stacked.reshape(-1, num_latent)
    joint_probs = jnp.prod(flattened, axis=1)

    return joint_probs

def multiplicative_cascade(num_latent:int, uncond_term:float, marg_support:jnp.ndarray)->jnp.ndarray:
    """Initial distribution given by the square root of the product of the marginal states 
    and a positive variable."""
    grids = jnp.meshgrid(*([marg_support] * num_latent), indexing='ij')
    stacked = jnp.stack(grids, axis=-1)
    flattened = stacked.reshape(-1, num_latent)
    prod = jnp.prod(flattened, axis=1)
    return uncond_term * jnp.sqrt(prod)