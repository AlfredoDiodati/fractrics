from fractrics.solvers import nelder_mead
from fractrics._components.HMM.base import hmm_metadata
from fractrics._components.HMM.forward.factor import update
from fractrics._components.HMM.data_likelihood import log_likelihood
from fractrics._components.HMM.transition_tensor import poisson_arrivals
from fractrics._components.HMM.initial_distribution import check_marg_prob_mass, multiplicative_cascade, factor_pmas

from dataclasses import dataclass, field, replace
from jax.lax import scan
from jax import hessian, jacrev, vmap

import jax.numpy as jnp
import jax.random as random
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
    data_likelihood = log_likelihood(mult_comp, states_values=latent_states)
    transition_tensor = poisson_arrivals(marg_prob_mass=marg_prob_mass, arrival_gdistance=arrival_gdistance, hf_arrival=hf_arrival, num_latent=self.num_latent)
    ergotic_dist = factor_pmas(marg_prob_mass, self.num_latent)
    NLL, current_distribution, distribution_list, nll_list = update(ergotic_dist, data_likelihood, transition_tensor)
    
    return replace(self, 
        optimization_info = {'negative_log_likelihood': NLL/(self.data[:-1]**variance_elasticity),
        'nll_list': nll_list},
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
        marginal_value = 1.0 + sigmoid(params_dict['marginal_value'])
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
            'marginal_value': inv_sigmoid(params_dict['marginal_value'] - 1.0)
            }
    
    unconstr_params = unconstrain_map(self.parameters)
    param_array, ravel_f = ravel_pytree(unconstr_params)
    marg_prob_mass = jnp.full(2, 0.5)
    
    def nll_f(prms:jnp.ndarray):
        
        frctl_component = multifractal_component(self)
        marg_support = jnp.array([prms[3], 2 - prms[3]])
        latent_states = multiplicative_cascade(num_latent=self.num_latent, uncond_term=prms[0], marg_support=marg_support)
        data_likelihood = log_likelihood(frctl_component, states_values=latent_states)
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
    
    H_total = hessian(objective_fun)(params_optimized)
    
    def constrain_map_se(params_dict):
        eps = 1e-6
        uncond_term = softplus(params_dict['unconditional_term'])
        arrival_gdistance = softplus(params_dict['arrival_gdistance']) + 1.0
        hf_arrival = jnp.clip(sigmoid(params_dict['hf_arrival']), eps, 1 - eps)
        marginal_value = 1.0 + jnp.clip(sigmoid(params_dict['marginal_value']), eps, 1 - eps)
        return jnp.array([uncond_term, arrival_gdistance, hf_arrival, marginal_value])

    def score_eta(params):
        _, nll_list = nll_f(constrain_map(ravel_f(params)))
        return nll_list
    
    G = jacrev(lambda x: constrain_map_se(ravel_f(x)))(params_optimized)
    S = jacrev(score_eta)(params_optimized)

    T = self.data_log_change.shape[0]
    A = (S.T @ S) / T
    cov_eta = jnp.linalg.inv(A) / T
    cov_theta = G @ cov_eta @ G.T
    standard_errors = jnp.sqrt(jnp.diag(cov_theta))
    
    def newey_west(S, L):
        T = S.shape[0]
        B = (S.T @ S) / T
        for l in range(1, L + 1):
            w = 1.0 - l / (L + 1.0)
            Gamma = (S[l:].T @ S[:-l]) / T
            B = B + w * (Gamma + Gamma.T)
        return B
    

    L = int(jnp.floor(4 * (T / 100.0) ** (2 / 9)))
    B_eta = newey_west(S, L)

    H_bar = H_total / T
    V_eta = jnp.linalg.inv(H_bar) @ B_eta @ jnp.linalg.inv(H_bar) / T
    V_theta_robust = G @ V_eta @ G.T
    robust_se = jnp.sqrt(jnp.diag(V_theta_robust))
    
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

def simulation(n_simulations: int, model_info: metadata, seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = random.PRNGKey(seed)
    key, key_init = random.split(key)

    m = model_info.parameters["marginal_value"]
    marginal_support = jnp.array([2.0 - m, m])
    marg_prob_mass = jnp.full(2, 0.5)

    latent_states = random.choice(
        key_init,
        marginal_support,
        (model_info.num_latent,),
        p=marg_prob_mass
    )

    keys = random.split(key, n_simulations * 3).reshape(n_simulations, 3, 2)
    poisson_arrivals = model_info._poisson_arrivals

    bernoulli_v = vmap(lambda k, p: random.bernoulli(k, p))
    choice_v = vmap(lambda k: random.choice(k, marginal_support, (), p=marg_prob_mass))

    intercept = model_info.parameters["intercept"]
    ar_coefficient = model_info.parameters["ar_coefficient"]
    variance_elasticity = model_info.parameters["variance_elasticity"]
    unconditional_term = model_info.parameters["unconditional_term"]

    initial_level = model_info.data[0]

    def _step(carry, key_triple):
        previous_level, states = carry
        key_arrival, key_switch, key_noise = key_triple

        arrival_keys = random.split(key_arrival, model_info.num_latent)
        switch_mask = bernoulli_v(arrival_keys, poisson_arrivals)

        switch_keys = random.split(key_switch, model_info.num_latent)
        new_states = choice_v(switch_keys)

        states = jnp.where(switch_mask, new_states, states)

        msm_volatility = unconditional_term * jnp.sqrt(jnp.prod(states))
        innovation = msm_volatility * random.normal(key_noise)

        level_scale = previous_level ** variance_elasticity
        increment = intercept + ar_coefficient * previous_level + level_scale * innovation
        current_level = previous_level + increment

        return (current_level, states), (current_level, msm_volatility, increment)

    (_, _), (level_sim, volatility_sim, return_sim) = scan(
        _step,
        (initial_level, latent_states),
        keys
    )

    return return_sim, volatility_sim, level_sim