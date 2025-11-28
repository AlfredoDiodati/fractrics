import jax.numpy as jnp
from jax.lax import scan
from jax.scipy.special import logsumexp

def joint_predictive(log_pi, log_A):
    """Joint transition step in log form"""
    return logsumexp(log_pi[:, None] + log_A, axis=0)

def make_predictive_function(log_transition_matrix)-> callable:
    """Returns a function that computes the predictive distribution"""
    def make_predictive_distribution(log_prior:jnp.ndarray)->jnp.ndarray:
        return logsumexp(log_prior[:, None] + log_transition_matrix, axis=0)
    return make_predictive_distribution

def update(distr_initial: jnp.ndarray,
            log_data_likelihood: jnp.ndarray,
            log_transition_matrix: jnp.ndarray
            )-> tuple:
    
    predictive_function = make_predictive_function(log_transition_matrix)
    log_distr_initial = jnp.log(distr_initial)
   
    def step(carry, log_data_likelihood_row):
        log_prior, nl_loss_likelihood = carry
        
        log_predictive_distribution = predictive_function(log_prior)
        
        log_nonnormalized_posterior = log_predictive_distribution + log_data_likelihood_row
        log_loss_likelihood = -logsumexp(log_nonnormalized_posterior)
        log_normalized_posterior = log_nonnormalized_posterior + log_loss_likelihood
        
        return (log_normalized_posterior, nl_loss_likelihood + log_loss_likelihood), (log_normalized_posterior, log_loss_likelihood)
    
    carry_initial = (log_distr_initial, 0.0)
    
    (log_final_posterior, final_loss), (log_distribution_list, nll_list) = scan(step, carry_initial, log_data_likelihood)
    return final_loss, jnp.exp(log_final_posterior), jnp.exp(log_distribution_list), nll_list

def pforecast(horizon:int, prior: jnp.ndarray, log_transition_matrix:jnp.ndarray):
    predictive_function = make_predictive_function(log_transition_matrix)
    
    def step(carry, _):
        carry = predictive_function(carry)
        return carry, carry
    _, predictive_list = scan(step, prior, xs=None, length=horizon)
    
    return predictive_list