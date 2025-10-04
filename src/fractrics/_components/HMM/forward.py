import jax.numpy as jnp
from jax.lax import scan
from jax.nn import softmax
from jax.scipy.special import logsumexp

def joint_predictive(log_pi, log_A):
    """Joint transition step in log form"""
    return logsumexp(log_pi[:, None] + log_A, axis=0)

def make_factor_predictive_function(*transition_matrices)-> callable:
    """Returns a function that computes the predictive distribution for independent states."""
    
    def make_predictive_distribution(prior:jnp.ndarray)->jnp.ndarray:
        
        dims = tuple(A.shape[0] for A in transition_matrices)
        predictive_tensor = prior.reshape(*dims)
        for axis, A in enumerate(transition_matrices):
            predictive_tensor = jnp.moveaxis(predictive_tensor, axis, -1)
            predictive_tensor = jnp.tensordot(predictive_tensor, A, axes=([-1], [0]))
            predictive_tensor = jnp.moveaxis(predictive_tensor, -1, axis)
        return predictive_tensor

    return make_predictive_distribution

def update(distr_initial: jnp.ndarray,
            data_likelihood: jnp.ndarray,
            transition_matrices: tuple[jnp.ndarray]
            )-> tuple:
    
    predictive_function = make_factor_predictive_function(*transition_matrices)
    
    small_constant = 1e-12
    
    dims = tuple(A.shape[0] for A in transition_matrices)
    log_tensor_initial_distribution = jnp.log(distr_initial.reshape(*dims) + small_constant)
    
    log_data_likelihood_tensor = jnp.log(data_likelihood.reshape((data_likelihood.shape[0],) + dims) + small_constant)
    
    #NOTE: predictive_tensor needs to run in non-log space
    #TODO: manage the forward with less swithches between log and non log space
    #TODO: create naive transition kernell and check with version is computationally more efficient f.e. use case
    
    def step(carry, log_data_likelihood_row):
        log_prior, nl_loss_likelihood = carry
        
        log_predictive_tensor = jnp.log(predictive_function(softmax(log_prior)) + small_constant)
        
        log_nonnormalized_posterior = log_predictive_tensor + log_data_likelihood_row
        log_loss_likelihood = -logsumexp(log_nonnormalized_posterior)
        log_normalized_posterior = log_nonnormalized_posterior + log_loss_likelihood
        
        return (log_normalized_posterior, nl_loss_likelihood + log_loss_likelihood), (log_normalized_posterior, log_loss_likelihood)
    
    carry_initial = (log_tensor_initial_distribution, 0.0)
    
    (log_final_posterior, final_loss), (log_distribution_list, nll_list) = scan(step, carry_initial, log_data_likelihood_tensor)
    # NOTE: nll_list is necessary for computing the robust standard errors
    return final_loss, jnp.exp(log_final_posterior), jnp.exp(log_distribution_list), nll_list

def pforecast(horizon:int, prior: jnp.ndarray, *transition_matrices: tuple[jnp.ndarray]):
    predictive_function = make_factor_predictive_function(*transition_matrices)
    
    def step(carry, _):
        carry = predictive_function(carry)
        return carry, carry
    _, predictive_list = scan(step, prior, xs=None, length=horizon)
    
    return predictive_list