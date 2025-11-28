from jax import lax
import jax.numpy as jnp
import jax.scipy.stats as jss
from jax.scipy.special import logsumexp
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass, replace, field

@dataclass(frozen=True)
class metadata():
    data: jnp.ndarray | None = None
    n_components : int = 2
    parameters : dict = field(default_factory = lambda: {
        'means': jnp.zeros(1),
        'variances':  jnp.ones(1),
        'probabilities': jnp.full(0.5)
    })
    
#TODO: generalize for distributions with different parameters
#TODO: generalize to multivariate case
# Note: works directly for mixture regression also
def _weights(data:jnp.ndarray, p:jnp.ndarray, mean:jnp.ndarray, scale:jnp.ndarray)->jnp.ndarray:
    """Computes the responsibility matrix for gaussian mixtures for each time point"""
    log_dens = jss.norm.logpdf(data[:, None],
                    loc=mean[None, :],
                    scale=scale[None, :])
    
    log_num = jnp.log(p)[None, :] + log_dens
    log_den = logsumexp(log_num, axis=1, keepdims=True)
    
    return jnp.exp(log_num - log_den)

def _Estep(data: jnp.ndarry, weights:jnp.ndarray, new_scales:jnp.ndarray,
    new_means:jnp.ndarray, new_probabilities: jnp.ndarray)->float:
    """The 'new' inputs are what is optimized in the M step.
    Used in fitting to diagnose correct optimization.
    """
    log_p   = jnp.log(new_probabilities + 1e-12)
    log_lik = jss.norm.logpdf(data[:, None], new_means[None,:], new_scales[None,:])
    return jnp.sum(weights * (log_p[None,:] + log_lik))

def _Mstep(data: jnp.ndarry, weights:jnp.ndarray):
    """
    Closed form M-step for linear gaussian mixtures.
    """
    wsum = jnp.sum(weights, axis=0)
    new_prob = wsum/data.shape[0]
    new_mean = (weights.T @ data) / wsum
    diff2 = (data[:, None] - new_mean[None, :]) ** 2
    new_scale = jnp.sum(weights * diff2, axis=0)/wsum 
    
    return new_prob, new_scale, new_mean
    