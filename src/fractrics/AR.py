
from fractrics._components.core import ts_metadata
from dataclasses import dataclass, replace

import jax.numpy as jnp

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
    noise: 