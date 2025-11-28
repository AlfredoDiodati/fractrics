from jax import lax
import jax.numpy as jnp
from types import ModuleType
import jax.scipy.stats as jss
from fractrics.solvers import nelder_mead
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass, replace, field
from fractrics._components.core import ts_metadata

@dataclass(frozen=True)
class metadata(ts_metadata):
    """
    General class for linear autoregressive models.
    
    The noise attribute determines which type of noise is used.
    Both OLS and ML estimations are implemented, altough:
    
    - For Gaussian AR the ML estimator is equivalent to OLS, so the latter is forced for numerical efficiency.
    - For other parametric noises, the ML estimator is more efficient (the variance of the parameters is lower than OLS)
    - Currently, the nonparametric noise needs to be estimated using OLS
    
    """
    data: jnp.ndarray | None = None
    noise: ModuleType | None = jss.norm
    parameters : dict = field(default_factory = lambda: {
        'coefficient': 0.5,
        'gaussian_variance':  1.0,
        'robust_variance': 1.0
    })
    squared_error : jnp.ndarray | None = None
    hyperparameters = None
    estimator: str = 'OLS'
    
def fit(model: metadata)-> metadata:

    is_OLS = model.estimator == 'OLS' or model.noise == jss.norm or model.noise is None
    data = model.data
    params_array, ravel_f = ravel_pytree(model.parameters)
    
    #TODO: return all relevant stuff in both cases
    
    def OLS(_):
        coefficient = jnp.sum(data[:-1]*data[1:])/jnp.sum(data[:-1]**2)
        error = (data[1:]-data[:-1]*coefficient)
        squared_error = error**2
        gaussian_variance = jnp.sum(squared_error)/(data.shape[0]-1)
        robust_variance = jnp.sum(squared_error * data[:-1]**2)/jnp.sum(data[:-1]**2)**2
        return coefficient, robust_variance
    
    def ML(params):
        coefficient = params[0]
        variance = params[1]
        error = data[1:] - data[:-1] * coefficient
        ll = jnp.sum(model.noise.logpdf(x=error, loc=0.0, scale=variance))
        return ll, error
    
    def objective_f(params):
        ll, _ = ML(params)
        return -ll
    
    def ML_estimation(_):
        params, nll, is_converged, num_iter = nelder_mead.solver(initial_guess=params_array, f=objective_f)
        _, error = ML(params)
        return params[0], params[1]
    
    coefficient, variance = lax.cond(is_OLS, OLS, ML_estimation, operand=None)
    
    return replace(model, parameters={'coefficient':coefficient, 'robust_variance': variance})
    
if __name__ == '__main__' and __debug__:
    
    import numpy as np
    from fractrics.utilities import summary
    noise = np.random.normal(0, 1, 100)
    coefficient = 0.5
    x = np.cumsum(noise * coefficient ** np.arange(100))
    
    model = metadata(data=x)
    fitted = fit(model)
    summary(fitted)
     