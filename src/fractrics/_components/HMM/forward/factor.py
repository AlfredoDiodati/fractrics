import jax.numpy as jnp
from jax.lax import scan
from jax.nn import softmax
from jax.scipy.special import logsumexp
    
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
    
    dims = tuple(A.shape[0] for A in transition_matrices)
    log_tensor_initial_distribution = jnp.log(distr_initial.reshape(*dims)) #eps
    
    log_data_likelihood_tensor = data_likelihood.reshape((data_likelihood.shape[0],) + dims)
    
    #NOTE: predictive_tensor needs to run in non-log space
    #TODO: manage the forward with less swithches between log and non log space
   
    def step(carry, log_data_likelihood_row):
        log_prior, nl_loss_likelihood = carry
        
        Pi_pred = predictive_function(softmax(log_prior))
        log_Pi_pred = jnp.log(Pi_pred)

        log_nonnormalized = log_Pi_pred + log_data_likelihood_row

        ell_t = logsumexp(log_nonnormalized)
        log_Pi_t = log_nonnormalized - ell_t
        nll_t = -ell_t

        
        return (log_Pi_t, nl_loss_likelihood + nll_t), (log_Pi_t, nll_t)
    
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