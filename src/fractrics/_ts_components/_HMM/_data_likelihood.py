""" Module for the data likelihood/emissions of the HMM. """

import jax.numpy as jnp
import jax.scipy.stats as jss

from abc import abstractmethod

class data_likelihood:

    def __init__(self, ts: jnp.ndarray) -> None:
        self.ts = ts
    
    @abstractmethod
    def likelihood(self, latent_states: jnp.ndarray)->jnp.ndarray:
        self.latent_states = latent_states

#TODO: generalize to work for any distribution in jss

class dlk_normal(data_likelihood):
    def likelihood(self, latent_states: jnp.ndarray)->jnp.ndarray:
        super().likelihood(latent_states)
        return jss.norm.pdf(self.ts.reshape(-1, 1), loc=0, scale=self.latent_states)