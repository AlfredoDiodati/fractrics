# Mid-level classes for Hidden Markov Models

# TODO: 
# GENERALIZE discrete_distribution to any discrete with dependence adjustment factor
# handling pandas inputs (e.g. py_trees)

# NOTE:
# __init__ of the components sets the hyperparameters, the specific functions take parameters as input (not stored as self instances)
# the abstract HMM calls the abstract methods in __init__, 
# ... the concretization should input the concrete version, along with the proper initialization inputs

import jax.numpy as jnp
import numpy as np

from abc import abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, field

from fractrics._ts_components._core import stochastic_ts, ts_metadata
import fractrics._ts_components._HMM._forward as _forward
import fractrics._ts_components._HMM._data_likelihood as _data_likelihood
import fractrics._ts_components._HMM._initial_distribution as _initial_distribution
import fractrics._ts_components._HMM._transition_tensor as _transition_tensor

@dataclass(frozen=True)
class hmm_metadata(ts_metadata):
    
    states_disjoint_MAP: jnp.ndarray | None = None
    states_viterbi: jnp.ndarray | None = None
    
    #TODO: consider enforcing typings inside dictionarieswith TypedDict
    filtered: Dict[str, Any] = field(default_factory=lambda:{
        'current_distribution': None,
        'distribution_list': None,
        'transition_tensor': None,
        'latent_states': None
    })
    
class HMM(stochastic_ts):
    """Generic class for Hidden Markov Models."""

    def __init__(self, ts: np.ndarray | jnp.ndarray,
                 forward: _forward.forward_update,
                 data_likelihood: _data_likelihood.data_likelihood,
                 initial_dist: _initial_distribution.homogeneous_independent,
                 transition_tensor: _transition_tensor.transition_tensor,
                 name: str | None = None,
                 num_latent: int = 1) -> None:
        
        super().__init__(ts, name)

        self.n_latent = num_latent
        self._initial_dist = initial_dist
        self._data_likelihood = data_likelihood
        self._transition_tensor = transition_tensor
        self._forward = forward
    
    def _MPM_states(self, states: jnp.ndarray, distr_list:jnp.ndarray)-> jnp.ndarray:
        """
        At each t returns the states corresponding to the maximized posterior disjoint marginal.
        distr_list: matrix where each column is the distribution of latent states at a given time, result of forward algorithm.
        """
        return states[jnp.argmax(distr_list, axis=1)]
    
    def _backward(self)-> jnp.ndarray:
        raise NotImplementedError
    
    def _viterbi(self)-> jnp.ndarray:
        """Computes the most probable path of latent states (joint probability)"""
        raise NotImplementedError

    # TODO: currently defined for the factor HMM, restructure for working in general (most of the code is the same)
    @abstractmethod
    def forecast(self, horizon):
        super().forecast(filtration=horizon)
