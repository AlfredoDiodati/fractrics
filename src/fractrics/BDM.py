import jax.numpy as jnp

from jax.scipy.linalg import toeplitz
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass, field, replace

from jax import hessian, jacrev
from fractrics.solvers import nelder_mead
from fractrics._components.core import ts_metadata

@dataclass(frozen=True)
class metadata(ts_metadata):
    data: jnp.ndarray | None = None
    parameters : dict = field(default_factory = lambda: {
        'Ksq':  None,
        'alpha': None
    })
    standard_error : dict = field(default_factory = lambda: {
        'Ksq':  None,
        'alpha': None
    })
    robust_standard_error : dict = field(default_factory = lambda: {
        'Ksq':  None,
        'alpha': None
    })
    
    sigma0: None
    se_sigma0: None
    rse_sigma0: None
    
    @property
    def data_log_change(self) -> jnp.ndarray:
        log_data = jnp.log(self.data)
        return log_data[1:] - log_data[:-1]
   
def fit(self:metadata, maxiter:int)->metadata:
    
    y = jnp.log(self.data_log_change())
    init_params, ravel_f = ravel_pytree(self.parameters)
    
    diff_matrix = toeplitz(jnp.arange(y.shape)) + 1.0
    
    def nll(params_input):
        par_dict = ravel_f(params_input)
        noise_mean = - par_dict['Ksq']*jnp.log(par_dict['alpha'])
        r = y - noise_mean
        log_sigma0 = jnp.mean(r)
        r = r - log_sigma0
        noise_covariance = - par_dict['Ksq'] * jnp.log(par_dict['alpha'] * diff_matrix)
        _, logdet = jnp.linalg.slogdet(noise_covariance)
        return 0.5 * (logdet + r.T @ jnp.linalg.solve(noise_covariance, r))
    
    params_opt, nll_v, is_converged, num_iteration = nelder_mead.solver(init_params,
    nll, maxiter)
    
    nll_hessian = hessian(nll)(params_opt)
    covariance = jnp.linalg.inv(nll_hessian)
    
    score_matrix = jacrev(nll)(params_opt)
    B = score_matrix.T @ score_matrix
    robust_covariance = covariance @ B @ covariance.T
    
    standard_errors = jnp.sqrt(jnp.diag(covariance))
    robust_se = jnp.sqrt(jnp.diag(robust_covariance))

    par_dict = ravel_f(params_opt)
    noise_mean = - par_dict['Ksq']*jnp.log(par_dict['alpha'])
    r = y - noise_mean
    sigma0 = jnp.exp(jnp.mean(r))

    n = y.shape
    e = jnp.ones(n)
    noise_covariance = - par_dict['Ksq'] * jnp.log(par_dict['alpha'] * diff_matrix)
    return replace(self, 
        parameters = par_dict,
        optimization_info = {
            'nll': nll_v,
            'num_iteration': num_iteration,
            'is_converged': is_converged 
        },
        standard_errors = ravel_f(standard_errors),
        robust_standard_errors = ravel_f(robust_se),
        sigma0 = sigma0,
        
        se_sigma0 = sigma0/(n**2)*(e.T @ noise_covariance @ e)
        )