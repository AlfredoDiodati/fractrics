"""Module of transition tensors"""

import jax.numpy as jnp
from abc import ABC, abstractmethod

# TODO: consider implementation with Tucker decomposition or other tensor decompositions (SVD/PCA)
#       ... to implement the independent and dependent transition kernels separately
#       ... similar concepts from Dynamic Bayesian Networks, exponential families

class transition_tensor(ABC):
    def __init__(self, num_latent:int)->None:
        self.num_latent = num_latent

    @abstractmethod
    def t_tensor(self)->jnp.ndarray:
        pass
    
class poisson_arrival(transition_tensor):
    """
    Transition happens with Poisson arrivals. State value is drawn by a marginal probability on states.
    The poisson arrivals are geometrically spaced.
    From Markov Switching Multifractal model. 
    """

    def __init__(self,
                num_latent:int,
                marg_prob_mass:jnp.ndarray
                ) -> None:
        super().__init__(num_latent)
        self.marg_prob_mass = marg_prob_mass

    #NOTE: currently the forward algorithm breaks if given a jax tensor as input, due to the way enumerate handles the predictive update
    def t_tensor(self, arrival_gdistance:float, hf_arrival:float)->tuple:
        arrivals = 1 - (1 - hf_arrival) ** (1 / (arrival_gdistance ** (jnp.arange(self.num_latent, 0, -1) - 1)))
        len_pm = len(self.marg_prob_mass)
        return tuple((1-g)*jnp.eye(len_pm) + g*jnp.tile(self.marg_prob_mass, (len_pm, 1)) for g in arrivals)