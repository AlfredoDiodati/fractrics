"""Module of transition tensors"""

import jax.numpy as jnp
# TODO: consider implementation with Tucker decomposition or other tensor decompositions (SVD/PCA)
#       ... to implement the independent and dependent transition kernels separately
#       ... similar concepts from Dynamic Bayesian Networks, exponential families
#TODO: change the for with a jax loop

#NOTE: currently the forward algorithm breaks if given a jax tensor as input, due to the way enumerate handles the predictive update
def poisson_arrivals(marg_prob_mass:jnp.ndarray, arrival_gdistance:float, hf_arrival:float, num_latent:int)->tuple:
    """
    Transition happens with Poisson arrivals. State value is drawn by a marginal probability on states.
    The poisson arrivals are geometrically spaced.
    From Markov Switching Multifractal model. 
    """
    k = jnp.arange(num_latent, 0, -1)
    arrivals = 1 - (1 - hf_arrival) ** (1 / (arrival_gdistance ** k))

    len_pm = marg_prob_mass.shape[0]

    A = (
        (1.0 - arrivals)[:, None, None] * jnp.eye(len_pm)[None, :, :]
        + arrivals[:, None, None] * marg_prob_mass[None, None, :]
    )

    return tuple(A)