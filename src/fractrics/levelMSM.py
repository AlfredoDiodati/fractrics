from fractrics import nelder_mead
from fractrics._components.HMM.base import hmm_metadata
from fractrics._components.HMM.forward import update
from fractrics._components.HMM.data_likelihood import likelihood
from fractrics._components.HMM.transition_tensor import poisson_arrivals
from fractrics._components.HMM.initial_distribution import check_marg_prob_mass, multiplicative_cascade, factor_pmas

from dataclasses import dataclass, field, replace
from jax import hessian, jacrev

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.nn import softplus, sigmoid

@dataclass(frozen=True)
class metadata(hmm_metadata):
    num_latent : int = 2
    nll_list: jnp.ndarray | None  = None
    parameters: dict = field(default_factory= lambda: {
        'intercept': None,
        'ar_coefficient': None,
        'variance_elasticity': None,
        'unconditional_term': None,
        'arrival_gdistance': None,
        'hf_arrival': None,
        'marginal_value': None
    })
    
    filtered: dict = field(default_factory= lambda: {
        'current_distribution': None,
        'distribution_list': None,
        'transition_tensor': None,
        'latent_states': None
    })
    
    standard_errors: dict = field(default_factory= lambda: {
        'intercept': None,
        'ar_coefficient': None,
        'variance_elasticity': None,
        'unconditional_term': None,
        'arrival_gdistance': None,
        'hf_arrival': None,
        'marginal_value': None
    })

    robust_standard_errors: dict = field(default_factory= lambda: {
        'intercept': None,
        'ar_coefficient': None,
        'variance_elasticity': None,
        'unconditional_term': None,
        'arrival_gdistance': None,
        'hf_arrival': None,
        'marginal_value': None
    })

    optimization_info : dict = field(default_factory= lambda: {
        'negative_log_likelihood': None,
        'is_converged': None,
        'n_iteration': None
    })
    
    @property
    def data_log_change(self) -> jnp.ndarray:
        log_data = jnp.log(self.data)
        return log_data[1:] - log_data[:-1]

    @property
    def _poisson_arrivals(self) -> jnp.ndarray:
        hf_arrival = self.parameters['hf_arrival']
        arrival_gdistance = self.parameters['arrival_gdistance']
        return 1 - (1 - hf_arrival) ** (1 / (arrival_gdistance ** (jnp.arange(self.num_latent, 0, -1) - 1)))
    
    @property
    def _MAP_disjoined(self) -> jnp.ndarray:
        return self.filtered['latent_states'][jnp.argmax(self.filtered['distribution_list'], axis=1)]

def multifractal_component(self:metadata)->jnp.ndarray:
    r = self.data_log_change
    return (r[1:]-r[:-1]*(1+self.parameters['ar_coefficient'])-self.parameters['intercept'])/(r[:-1]**self.parameters['variance_elasticity'])
  
def filter(self:metadata) -> None:
    variance_elasticity = self.parameters['variance_elasticity']
    mult_comp = multifractal_component(self)
    uncond_term = self.parameters['unconditional_term']
    arrival_gdistance = self.parameters['arrival_gdistance']
    hf_arrival = self.parameters['hf_arrival']
    m0 = self.parameters['marginal_value']
    marg_support = jnp.array([m0, 2 - m0])
    marg_prob_mass = jnp.full(2, 0.5)
    
    latent_states = multiplicative_cascade(num_latent=self.num_latent, uncond_term=uncond_term, marg_support=marg_support)
    data_likelihood = likelihood(mult_comp, states_values=latent_states)
    transition_tensor = poisson_arrivals(marg_prob_mass=marg_prob_mass, arrival_gdistance=arrival_gdistance, hf_arrival=hf_arrival, num_latent=self.num_latent)
    ergotic_dist = factor_pmas(marg_prob_mass, self.num_latent)
    NLL, current_distribution, distribution_list, nll_list = update(ergotic_dist, data_likelihood, transition_tensor)
    
    return replace(self, 
        optimization_info = {'negative_log_likelihood': NLL/(self.data[:-1]**variance_elasticity)},
        filtered = {
        'current_distribution': current_distribution,
        'distribution_list': distribution_list,
        'transition_tensor': transition_tensor,
        'latent_states': latent_states,
        'multifractal_component': mult_comp
        }
    )
    
def fit(self:metadata, max_iter:int):
    
    def constrain_map(params_dict:dict) -> jnp.ndarray:
        variance_elasticity = softplus(param_dict['variance_elasticity'])
        uncond_term=softplus(params_dict['unconditional_term'])
        arrival_gdistance=softplus(params_dict['arrival_gdistance']) + 1
        hf_arrival=sigmoid(params_dict['hf_arrival'])
        marginal_value = softplus(params_dict['marginal_value'])
        params_array = jnp.array([uncond_term, arrival_gdistance, 
            hf_arrival, marginal_value, variance_elasticity, 
            param_dict['intercept'], param_dict['ar_coefficient']])
        return params_array
    
    def unconstrain_map(params_dict:dict) -> dict:
        """
        Used to let the initialized parameters be used directly in optimization,
        otherwise they would be changed by constrain map.
        """
        def inv_softplus(y): return jnp.log(jnp.expm1(y))
        def inv_sigmoid(y): return jnp.log(y / (1.0 - y))
        return{
            'intercept': param_dict['intercept'],
            'ar_coefficient': param_dict['ar_coefficient'],
            'variance_elasticity': inv_softplus(param_dict['variance_elasticity']),
            'unconditional_term': inv_softplus(params_dict['unconditional_term']),
            'arrival_gdistance': inv_softplus(params_dict['arrival_gdistance']- 1),
            'hf_arrival': inv_sigmoid(params_dict['hf_arrival']),
            'marginal_value': inv_softplus(params_dict['marginal_value'])
            }
    
    unconstr_params = unconstrain_map(self.parameters)
    param_array, ravel_f = ravel_pytree(unconstr_params)
    marg_prob_mass = jnp.full(2, 0.5)
    
    def nll_f(prms:jnp.ndarray):
        
        frctl_component = multifractal_component(self)
        marg_support = jnp.array([prms[3], 2 - prms[3]])
        latent_states = multiplicative_cascade(num_latent=self.num_latent, uncond_term=prms[0], marg_support=marg_support)
        data_likelihood = likelihood(frctl_component, states_values=latent_states)
        transition_tensor = poisson_arrivals(marg_prob_mass=marg_prob_mass, arrival_gdistance=prms[1],
            hf_arrival=prms[2], num_latent=self.num_latent)
        NLL, _, _, nll_list = update(ergotic_dist, data_likelihood, transition_tensor)
        return NLL/(self.data[:-1]**prms[4]), nll_list

    def objective_fun(params:jnp.ndarray):
        """Negative log likelihood with constrains."""
        param_dict = ravel_f(params)
        prms = constrain_map(param_dict)
        NLL, _ = nll_f(prms)
        return NLL

    check_marg_prob_mass(marg_prob_mass)
    ergotic_dist = factor_pmas(marg_prob_mass, self.num_latent)
        
    params_optimized, nll, is_converged, num_iterations = nelder_mead.solver(initial_guess=param_array, f=objective_fun, max_iter=max_iter)
    
    param_dict = ravel_f(params_optimized)
    prms = constrain_map(param_dict)
    nll_hessian,_ = hessian(nll_f, has_aux=True)(prms)
    
    covariance = jnp.linalg.inv(nll_hessian)
    standard_errors = jnp.sqrt(jnp.diag(covariance))
    
    def score_fun(prms:jnp.ndarray):
        _, nll_list = nll_f(prms)
        return nll_list
    
    score_matrix = jacrev(score_fun)(prms)
    B = score_matrix.T @ score_matrix
    robust_covariance = covariance @ B @ covariance.T
    robust_se = jnp.sqrt(jnp.diag(robust_covariance))
    
    fit_metadata = replace(self,
        
        parameters = {
        'intercept': prms[6],
        'ar_coefficient': prms[5],
        'variance_elasticity': prms[4],
        'unconditional_term': prms[0],
        'arrival_gdistance': prms[1],
        'hf_arrival': prms[2],
        'marginal_value': prms[3]
        },
        
        optimization_info = {
            'negative_log_likelihood': nll,
            'n_iteration': num_iterations,
            'is_converged': is_converged
        },
        standard_errors = {
            'intercept': standard_errors[6],
            'ar_coefficient': standard_errors[5],
            'variance_elasticity': standard_errors[4],
            'unconditional_term': standard_errors[0],
            'arrival_gdistance': standard_errors[1],
            'hf_arrival': standard_errors[2],
            'marginal_value': standard_errors[3]
        },
        robust_standard_errors = {
            'intercept': robust_se[6],
            'ar_coefficient': robust_se[5],
            'variance_elasticity': robust_se[4],
            'unconditional_term': robust_se[0],
            'arrival_gdistance': robust_se[1],
            'hf_arrival': robust_se[2],
            'marginal_value': robust_se[3]
        }
    )
    return fit_metadata