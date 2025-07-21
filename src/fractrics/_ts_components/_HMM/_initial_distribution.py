""" Module for the initial distribution of the HMM. """

import jax.numpy as jnp
import itertools

from abc import abstractmethod

class initial_distr:
    """
    The initial distribution of the process. Either the ergotic distribution, 
    or another initial choice based on context-criteria.
        
    It can also be learned from data if it depends on parameters.
    """
    def __init__(self, marg_prob_mass:jnp.ndarray, num_latent:int= 1) -> None:
        
        if jnp.isclose(jnp.sum(marg_prob_mass), 1.0, atol=1e-6):
            self.num_latent = num_latent
            self.marg_prob_mass = marg_prob_mass
        else: raise ValueError("The marginal probability mass needs to sum to 1.")

# TODO: add generalized discrete distribution function with dependence function.

class homogeneous_independent(initial_distr):
    """
    Class for initial joint distribution of independent latent states with the same marginal distribution (homogeneous).
    """
    
    @abstractmethod
    def support(self, marg_support:jnp.ndarray) -> jnp.ndarray:
        """
        Joint support of the distribution.
        At this level, only the disjoint states are computed. The aggregation type is model dependent.
        """
        self.support_disjoint = jnp.array(list(itertools.product(marg_support, repeat=self.num_latent)))
    
    def mass(self) -> jnp.ndarray:
        """
        Joint initial probability mass. The joint probabilities are the row-wise product of the cartesian product 
        repeated as the number of latent states - the dependece term (implemented in concretization classes).
        
        Returns it in a tensor form, ready for tensor-forward algorithm.
        """
        prob_disjoint = jnp.array(list(itertools.product(self.marg_prob_mass, repeat=self.num_latent)))
        return jnp.prod(prob_disjoint, axis=1)
    
class multiplicative_cascade(homogeneous_independent):
    """
    Initial distribution given by the square root of the product of the marginal states and a positive variable.
    """
    def support(self, uncond_term:float, marg_support:jnp.ndarray)->jnp.ndarray:
        super().support(marg_support)
        return uncond_term * jnp.sqrt(jnp.prod(self.support_disjoint, axis=1))