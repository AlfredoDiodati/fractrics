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
    arrivals = 1 - (1 - hf_arrival) ** (1 / (arrival_gdistance ** (jnp.arange(num_latent, 0, -1) - 1)))
    len_pm = len(marg_prob_mass)
    return tuple((1-g)*jnp.eye(len_pm) + g*jnp.tile(marg_prob_mass, (len_pm, 1)) for g in arrivals)