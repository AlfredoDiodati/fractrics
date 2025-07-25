
from abc import abstractmethod

import jax.numpy as jnp
from jax.lax import scan
from jax.nn import softmax
from jax.scipy.special import logsumexp

# TODO: add touple check using from typing import Tuple
# consider precision tuning (float32/bfloat16) for speedup
# build optimized engine for HMMs where emission factorize
# consider using predictive update directly in log space

class forward_update:
    """
    Abstract forward algorithm for discrete time HMMs. Inputs are assumed to not already be in log form.
    Distinct from continuous latent models, in which probability distributions cannot be represented using tensors.
    """
    def __init__(self,
                 num_latent: int = 1) -> None:
        """
        num_latent: number of latent variables
        num_states: number of states that each latent (homogeneous) variable can assume
        """

        self.num_latent = num_latent
    
    @abstractmethod
    def make_predictive_function(self):
        """Function that defines and returns the function for computing the predictive tensor."""
        pass
    
    @abstractmethod
    def update(self,
               distr_initial: jnp.ndarray,
               data_likelihood: jnp.ndarray,
               trans_tensor: jnp.ndarray
               ):
        """Function that updates the latent distribution along a matrix of emissions. Should use the function of make_predictive_function"""
        pass

class factor_transition(forward_update):
    """
    Forward algorithm for discrete time HMM with factorable latent states and joint emission.
    trans_tensor is supposed to be such that each of its dimension is the marginal transition matrix of each latent state.
    """
    
    @staticmethod # necessary to be used in update
    def make_predictive_function(*transition_matrices):
        """Returns a function that computes the predictive distribution in tensor form."""
        
        def make_predictive_distribution(prior:jnp.ndarray)->jnp.ndarray:
            
            dims = tuple(A.shape[0] for A in transition_matrices)
            predictive_tensor = prior.reshape(*dims)
            for axis, A in enumerate(transition_matrices):
                predictive_tensor = jnp.moveaxis(predictive_tensor, axis, -1)
                predictive_tensor = jnp.tensordot(predictive_tensor, A, axes=([-1], [0]))
                predictive_tensor = jnp.moveaxis(predictive_tensor, -1, axis)
            return predictive_tensor

        return make_predictive_distribution
    
    def update(self,
               distr_initial: jnp.ndarray,
               data_likelihood: jnp.ndarray,
               transition_matrices: tuple[jnp.ndarray]
              )-> tuple:
        
        predictive_function = self.make_predictive_function(*transition_matrices)
        
        small_constant = 1e-45
        
        dims = tuple(A.shape[0] for A in transition_matrices)
        log_tensor_initial_distribution = jnp.log(distr_initial.reshape(*dims) + small_constant)
        
        log_data_likelihood_tensor = jnp.log(data_likelihood.reshape((data_likelihood.shape[0],) + dims) + small_constant)
        
        #NOTE: predictive_tensor needs to run in non-log space
        #TODO: manage the forward with less swithches between log and non log space
        def step(carry, log_data_likelihood_row):
            log_prior, nl_loss_likelihood = carry
            
            log_predictive_tensor = jnp.log(predictive_function(softmax(log_prior)) + small_constant)
            
            log_nonnormalized_posterior = log_predictive_tensor + log_data_likelihood_row
            log_loss_likelihood = -logsumexp(log_nonnormalized_posterior)
            log_normalized_posterior = log_nonnormalized_posterior + log_loss_likelihood
            return (log_normalized_posterior, nl_loss_likelihood + log_loss_likelihood), log_normalized_posterior
        
        carry_initial = (log_tensor_initial_distribution, 0.0)
        
        (log_final_posterior, final_loss), log_distribution_list = scan(step, carry_initial, log_data_likelihood_tensor)
        return final_loss, jnp.exp(log_final_posterior), jnp.exp(log_distribution_list)
    
    #TODO: move in HMM base class by handling the specific inputs
    def forecast(self, horizon:int, prior: jnp.ndarray, *transition_matrices: jnp.ndarray):
        predictive_function = self.make_predictive_function(*transition_matrices)
        
        def step(carry, _):
            carry = predictive_function(carry)
            return carry, carry
        _, predictive_list = scan(step, prior, xs=None, length=horizon)
        
        return predictive_list