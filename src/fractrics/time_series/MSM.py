from fractrics._ts_components._HMM._base import HMM, hmm_metadata
import fractrics._ts_components._HMM._forward as _forward
import fractrics._ts_components._HMM._transition_tensor as _transition_tensor
import fractrics._ts_components._HMM._data_likelihood as _data_likelihood
import fractrics._ts_components._HMM._initial_distribution as _initial_distribution

from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Any

from jax.lax import scan
from jax.nn import softplus, sigmoid
from jax import jit #,hessian, jacfwd, vmap # for standard errors

import numpy as np

import jax.numpy as jnp
import jax.random as random

from jaxopt import BFGS

@dataclass(frozen=True)
class msm_metadata(hmm_metadata):
    
    data_log_change: jnp.ndarray | None = None
    _poisson_arrivals: jnp.ndarray = field(default_factory=lambda: jnp.full(1, 0.0))
    
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        'unconditional_term': None,
        'arrival_gdistance': None,
        'hf_arrival': None,
        'marginal_support': None
    })
    
    hyperparameters : Dict[str, Any] = field(default_factory=lambda: {
        'n_latent': None,
        'marginal_probability_mass': None
    })
    
    optimization_info : Dict[str, Any] = field(default_factory=lambda: {
        'loss_gradient': None,
        'n_iteration': None
    })

class MSM(HMM):
    """Univariate Discrete Markov Switching Multifractal model."""
    
    def __init__(self, ts: np.ndarray | jnp.ndarray,
                 n_latent: int = 1,
                 marg_prob_mass = jnp.full(2, 0.5),
                 name: str | None = None,
        ) -> None:
        
        if (jnp.any(ts <= 0 | jnp.isinf(ts))): raise ValueError("MSM is defined for positive, finite time series only.")
        else:
            self.r = jnp.log(ts[1:])-jnp.log(ts[:-1])
            self.marg_prob_mass = marg_prob_mass #necessary to correctly create optimization constrains

            super().__init__(ts=ts,
                num_latent=n_latent,
                initial_dist= _initial_distribution.multiplicative_cascade(num_latent=n_latent, marg_prob_mass=marg_prob_mass),
                transition_tensor = _transition_tensor.poisson_arrival(num_latent=n_latent, marg_prob_mass=marg_prob_mass),
                data_likelihood=_data_likelihood.dlk_normal(ts=self.r),
                forward=_forward.factor_transition(num_latent=n_latent),
                name=name)

    def fit(self, initial_parameters, maxiter:int, verbose=False):
        """
        Parameters to be optimized, stored in initial_parameters in the following order:
            uncond_term: unconditional variance component of the latent states: > 0
            arrival_gdistance: geometric distrance between each Poisson arrival: > 0
            hf_arrival: the high-frequency Poisson arrival: between 0 and 1
            marg_support: (rest of the parameters vector) the support of the marginal distribution: >0 and unity expectation.
        """
        
        ergotic_dist  = self._initial_dist.mass()
        
        def reparameterization(params):
            """Enforces constrains on parameters before input to the solver."""
            positive_constraint = softplus(params[:2])
            possion_constraint = sigmoid(params[2])
            support = params[3:]
            support_positive = jnp.exp(support)
            support_constraint = support_positive / jnp.dot(self.marg_prob_mass, support_positive)           
            return jnp.concatenate([positive_constraint, jnp.array([possion_constraint]), support_constraint]) # type: ignore
        
        def loss_fn(params):
            
            constrained_params = reparameterization(params)
            
            uncond_term=constrained_params[0]
            arrival_gdistance=constrained_params[1]
            hf_arrival=constrained_params[2]
            marg_support = constrained_params[3:]

            latent_states = self._initial_dist.support(uncond_term=uncond_term, marg_support=marg_support) # type: ignore
            data_likelihood = self._data_likelihood.likelihood(latent_states=latent_states)
            transition_tensor = self._transition_tensor.t_tensor(arrival_gdistance=arrival_gdistance, hf_arrival=hf_arrival) # type: ignore
            
            NLL, distr_fin, distr_list = self._forward.update(ergotic_dist, data_likelihood, transition_tensor) # type: ignore
            
            return NLL, (distr_fin, transition_tensor, distr_list, latent_states)

        solver = BFGS(fun=loss_fn, has_aux=True, maxiter=maxiter, verbose=verbose)
        result = solver.run(init_params=initial_parameters)
        
        params_optimized = reparameterization(result.params)
        
        fit_metadata = msm_metadata(
            data=self.ts,
            data_log_change = self.r,
            name= self.name,
            states_disjoint_MAP= self._MPM_states(result.state.aux[3], result.state.aux[2]),
            
            _poisson_arrivals = 1 - (1 - params_optimized[2]) ** (1 / (params_optimized[1] ** (jnp.arange(self.n_latent, 0, -1) - 1))),
            
            filtered = {
                'current_distribution': result.state.aux[0],
                'distribution_list': result.state.aux[2],
                'transition_tensor': result.state.aux[1],
                'latent_states': result.state.aux[3]
            },
            
            parameters = {
                'unconditional_term': params_optimized[0],
                'arrival_gdistance': params_optimized[1],
                'hf_arrival': params_optimized[2],
                'marginal_support': params_optimized[3:]
            },
            
            hyperparameters = {
                'n_latent': self.n_latent,
                'marginal_probability_mass': self.marg_prob_mass
            },
            
            optimization_info = {
                'negative_log_likelihood': result.state.value,
                'loss_gradient': result.state.grad,
                'n_iteration': result.state.iter_num
            }
        )

        #TODO:           
            #standard error (not tested):
                # nll_hessian = hessian(negative_log_likelihood)(params_optimized, self.r)
                # covariance_matrix = jnp.linalg.inv(nll_hessian)
                # standard_errors = jnp.sqrt(jnp.diag(covariance_matrix))
            #robust standard errors (not tested):
                # assume nll_observation is nll not summed (ony need to modify scan above)
                # observation_jacobian = jacfwd(nll_observation)
                # score_matrix = vmap(observation_jacobian, in_axes=(None, 0))(params_optimized, self.r)
                # outer_product_scores = jnp.dot(score_matrix.T, score_matrix)
                # robust_cov = jnp.dot(jnp.dot(covariance_matrix, outer_product_scores),covariance_matrix)
                # robust_se = jnp.sqrt(jnp.diag(robust_cov))
                                
        return fit_metadata
    
    def simulation(self,
            n_simulations:int,
            model_info:msm_metadata,
            key = random.PRNGKey(0))->tuple[jnp.ndarray, jnp.ndarray]:

        key, key_init = random.split(key)

        initial_states = random.choice(
            key_init, model_info.parameters['marginal_support'],
            (model_info.hyperparameters['n_latent'],), 
            p=model_info.hyperparameters['marginal_probability_mass']
            )
        initial_random_keys = random.split(key, n_simulations * 3).reshape(n_simulations, 3, 2)
        
        def _step(states, key_triple):

            key_arrival, key_switch, key_noise = key_triple
            switch_mask = random.bernoulli(key_arrival, p=model_info._poisson_arrivals)
            
            new_vals = random.choice(
                key_switch, model_info.parameters['marginal_support'],
                (model_info.hyperparameters['n_latent'],), 
                p=model_info.hyperparameters['marginal_probability_mass']
                )
            
            states = jnp.where(switch_mask, new_vals, states)
            vol = model_info.parameters['unconditional_term'] * jnp.sqrt(jnp.prod(states)) # type: ignore
            r = vol*random.normal(key_noise)
            
            return states, (vol, r)
        
        _, (volatility_sim, return_sim) = scan(_step,initial_states, initial_random_keys)
        
        return return_sim, volatility_sim
    
    #TODO: generalize in base HMM class
    def forecast(self, horizon:int, model_info: msm_metadata) -> jnp.ndarray:
        
        return self._forward.forecast(horizon,  #type: ignore
        model_info.filtered['current_distribution'],
        *model_info.filtered['transition_tensor'])