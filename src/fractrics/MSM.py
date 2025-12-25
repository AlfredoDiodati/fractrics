from fractrics.solvers import nelder_mead, momentum_gd
from fractrics._components.HMM.base import hmm_metadata
from fractrics._components.HMM.forward.factor import pforecast, update
from fractrics._components.HMM.transition_tensor import poisson_arrivals
from fractrics._components.HMM.data_likelihood import log_likelihood
from fractrics._components.HMM.initial_distribution import check_marg_prob_mass, multiplicative_cascade, factor_pmas

from dataclasses import dataclass, field, replace
from jax.lax import scan
from jax import hessian, jacrev, vmap

import jax.numpy as jnp
import jax.random as random
from jax.flatten_util import ravel_pytree
from jax.nn import softplus, sigmoid
#TODO: make a filter function that recovers the current states, MAP, Viterbi, ...

@dataclass(frozen=True)
class metadata(hmm_metadata):
    num_latent : int = 2
    nll_list: jnp.ndarray | None  = None
    parameters: dict = field(default_factory= lambda: {
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
        'unconditional_term': None,
        'arrival_gdistance': None,
        'hf_arrival': None,
        'marginal_value': None
    })

    robust_standard_errors: dict = field(default_factory= lambda: {
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
    
def filter(self:metadata) -> metadata:
    uncond_term = self.parameters['unconditional_term']
    arrival_gdistance = self.parameters['arrival_gdistance']
    hf_arrival = self.parameters['hf_arrival']
    m0 = self.parameters['marginal_value']
    marg_support = jnp.array([m0, 2 - m0])
    marg_prob_mass = jnp.full(2, 0.5)
    
    latent_states = multiplicative_cascade(num_latent=self.num_latent, uncond_term=uncond_term, marg_support=marg_support)
    data_likelihood = log_likelihood(self.data_log_change, states_values=latent_states)
    transition_tensor = poisson_arrivals(marg_prob_mass=marg_prob_mass, arrival_gdistance=arrival_gdistance, hf_arrival=hf_arrival, num_latent=self.num_latent)
    ergotic_dist = factor_pmas(marg_prob_mass, self.num_latent)
    NLL, current_distribution, distribution_list, nll_list = update(ergotic_dist, data_likelihood, transition_tensor)
    
    return replace(self, 
        optimization_info  =  {
        'negative_log_likelihood': NLL,
        'nll_list': nll_list
        },
        filtered = {
        'current_distribution': current_distribution,
        'distribution_list': distribution_list,
        'transition_tensor': transition_tensor,
        'latent_states': latent_states
        }
    )
    
def fit(self:metadata, max_iter:int, solver:str='nelder-mead'):
    """
    Supported solvers: 'nelder-mead' and 'momentum_gd'
    """
    if solver == 'nelder-mead': method = nelder_mead
    else: method = momentum_gd
    
    def constrain_map(params_dict:dict) -> jnp.ndarray:
        uncond_term=softplus(params_dict['unconditional_term'])
        arrival_gdistance=softplus(params_dict['arrival_gdistance']) + 1
        hf_arrival=sigmoid(params_dict['hf_arrival'])
        marginal_value = 1.0 + sigmoid(params_dict['marginal_value'])
        params_array = jnp.array([uncond_term, arrival_gdistance, hf_arrival, marginal_value])
        return params_array
    
    def unconstrain_map(params_dict:dict) -> dict:
        """
        Used to let the initialized parameters be used directly in optimization,
        otherwise they would be changed by constrain map.
        """
        def inv_softplus(y): return jnp.log(jnp.expm1(y))
        def inv_sigmoid(y): return jnp.log(y / (1.0 - y))
        return{
            'unconditional_term': inv_softplus(params_dict['unconditional_term']),
            'arrival_gdistance': inv_softplus(params_dict['arrival_gdistance']- 1),
            'hf_arrival': inv_sigmoid(params_dict['hf_arrival']),
            'marginal_value': inv_sigmoid(params_dict['marginal_value'] - 1.0)
            }
    
    unconstr_params = unconstrain_map(self.parameters)
    param_array, ravel_f = ravel_pytree(unconstr_params)
    marg_prob_mass = jnp.full(2, 0.5)
    
    def nll_f(prms:jnp.ndarray):
        marg_support = jnp.array([prms[3], 2 - prms[3]])
        latent_states = multiplicative_cascade(num_latent=self.num_latent, uncond_term=prms[0], marg_support=marg_support)
        data_likelihood = log_likelihood(self.data_log_change, states_values=latent_states)
        transition_tensor = poisson_arrivals(marg_prob_mass=marg_prob_mass, arrival_gdistance=prms[1],
            hf_arrival=prms[2], num_latent=self.num_latent)
        NLL, _, _, nll_list = update(ergotic_dist, data_likelihood, transition_tensor)
        return NLL, nll_list
        
    def objective_fun(params:jnp.ndarray):
        """Negative log likelihood with constrains."""
        param_dict = ravel_f(params)
        prms = constrain_map(param_dict)
        NLL, _ = nll_f(prms)
        return NLL

    check_marg_prob_mass(marg_prob_mass)
    ergotic_dist = factor_pmas(marg_prob_mass, self.num_latent)
        
    params_optimized, nll, is_converged, num_iterations = method.solver(initial_guess=param_array, f=objective_fun, max_iter=max_iter)
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
            'unconditional_term': standard_errors[0],
            'arrival_gdistance': standard_errors[1],
            'hf_arrival': standard_errors[2],
            'marginal_value': standard_errors[3]
        },
        robust_standard_errors = {
            'unconditional_term': robust_se[0],
            'arrival_gdistance': robust_se[1],
            'hf_arrival': robust_se[2],
            'marginal_value': robust_se[3]
        }
    )
    return fit_metadata

def simulation(n_simulations: int, model_info: metadata, seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = random.PRNGKey(seed)
    key, key_init = random.split(key)

    m = model_info.parameters["marginal_value"]
    marginal_support = jnp.array([2.0 - m, m])
    marg_prob_mass = jnp.full(2, 0.5)

    initial_states = random.choice(
        key_init,
        marginal_support,
        (model_info.num_latent,),
        p=marg_prob_mass
    )

    keys = random.split(key, n_simulations * 3).reshape(n_simulations, 3, 2)
    pa = model_info._poisson_arrivals  # shape (K,)

    bernoulli_v = vmap(lambda k, p: random.bernoulli(k, p))
    choice_v = vmap(lambda k: random.choice(k, marginal_support, (), p=marg_prob_mass))

    def _step(states, key_triple):
        key_arrival, key_switch, key_noise = key_triple

        key_arrivals = random.split(key_arrival, model_info.num_latent)
        switch_mask = bernoulli_v(key_arrivals, pa)

        key_switches = random.split(key_switch, model_info.num_latent)
        new_vals = choice_v(key_switches)

        states = jnp.where(switch_mask, new_vals, states)

        vol = model_info.parameters["unconditional_term"] * jnp.sqrt(jnp.prod(states))
        r = vol * random.normal(key_noise)

        return states, (vol, r)

    _, (volatility_sim, return_sim) = scan(_step, initial_states, keys)
    return return_sim, volatility_sim

# def simulation(n_simulations:int,
#         model_info:metadata,
#         seed:int=0)->tuple[jnp.ndarray, jnp.ndarray]:
    
#     key = random.PRNGKey(seed)
#     key, key_init = random.split(key)
#     marginal_support = jnp.array([2 - model_info.parameters['marginal_value'], model_info.parameters['marginal_value']])
#     marg_prob_mass = jnp.full(2, 0.5)
#     initial_states = random.choice(
#         key_init, marginal_support,
#         (model_info.num_latent,), 
#         p=marg_prob_mass)
    
#     initial_random_keys = random.split(key, n_simulations * 3).reshape(n_simulations, 3, 2)
#     pa = model_info._poisson_arrivals
    
#     def _step(states, key_triple):

#         key_arrival, key_switch, key_noise = key_triple
#         switch_mask = random.bernoulli(key_arrival, p=pa)
        
#         new_vals = random.choice(
#             key_switch, marginal_support,
#             (model_info.num_latent,), 
#             p=marg_prob_mass
#             )
        
#         states = jnp.where(switch_mask, new_vals, states)
#         vol = model_info.parameters['unconditional_term'] * jnp.sqrt(jnp.prod(states))
#         r = vol*random.normal(key_noise)
        
#         return states, (vol, r)
    
#     _, (volatility_sim, return_sim) = scan(_step,initial_states, initial_random_keys)
    
#     return return_sim, volatility_sim

def variance_forecast(horizon:int, model_info: metadata, 
    quantiles: tuple[float, float]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    quantiles = jnp.array(quantiles)
    prob_list = pforecast(horizon,model_info.filtered['current_distribution'],*model_info.filtered['transition_tensor'])
    uncond_term = model_info.parameters["unconditional_term"]**2
    state_values = model_info.filtered['latent_states']
    prob_flat = prob_list.reshape(prob_list.shape[0], -1)
    prod_state_values = state_values
    expected_variance = uncond_term*(prob_flat @ prod_state_values)
    
    def weighted_quantile(values, probs, quantiles):
        """Compute weighted quantiles for a 1D distribution."""
        sorter = jnp.argsort(values)
        values, probs = values[sorter], probs[sorter]
        cumprobs = jnp.cumsum(probs)
        return jnp.interp(quantiles, cumprobs, values)
    
    cis = vmap(lambda p: weighted_quantile(uncond_term*prod_state_values, p, quantiles))(prob_flat)
    ci_lower, ci_upper = cis[:,0], cis[:,1]
    return expected_variance, ci_lower, ci_upper

def boostrap_forecast(self:metadata, horizon:int, num_simulation:int, seed: int = 0):
    seeds = jnp.arange(seed, seed + num_simulation)
    returns_batch, vols_batch = vmap(lambda s: simulation(horizon, self, s))(seeds)
    return returns_batch, vols_batch