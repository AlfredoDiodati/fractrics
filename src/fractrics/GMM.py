from jax import lax
import jax.numpy as jnp
import jax.scipy.stats as jss
from jax.scipy.special import logsumexp, softmax
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
    optimization_info : dict = field(default_factory= lambda: {
        'log likelihood': None,
        'is_converged': None,
        'n_iteration': None
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

def _link(scale, prb):
    """Ensures scale is positive and probabilities sum to one"""
    scale = jnp.exp(scale)
    prb = softmax(prb)
    return scale, prb

def fit(model: metadata, tol:float=1e-8, max_iter: int = 1000):
    data = model.data
    init_p = model.parameters["probabilities"]
    init_mean = model.parameters["means"]
    init_scale = model.parameters["means"]
    
    def _step(carry):
        p, mean, scale, oldlik, it = carry
        scale, p = _link(scale, p)
        w = _weights(data, p, mean, scale)
        new_prob, new_scale, new_mean = _Mstep(data, w)
        newlik = _Estep(data, w, new_scale, new_mean)
        likratio = jnp.abs(newlik / oldlik - 1.0)
        it = it + 1
        return (new_prob, new_mean, new_scale, newlik, it), likratio
    
    def _stop(carry):
        _, _, _, oldlik, it = carry
        return jnp.logical_and(it < max_iter, jnp.isfinite(oldlik))

    init_w = _weights(data, init_p, init_mean, init_scale)
    init_lik = _Estep(data, init_w, init_scale, init_mean)
    likratio0 = jnp.inf
    it0 = 0
    init_carry = (init_p, init_mean, init_scale, init_lik, likratio0, it0)
    p, mean, scale, lik, _, it = lax.while_loop(_stop, _step, init_carry)
    
    return replace(model, 
        parameters = {
            'probabilities': p,
            'means': mean,
            'scale': scale
        },
        optimization_info = {
        'log likelihood': lik,
        'is_converged': False,
        'n_iteration': it
        }
    )