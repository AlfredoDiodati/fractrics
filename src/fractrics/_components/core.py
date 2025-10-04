# Contains abstract time series classes. 

import jax.numpy as jnp
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ts_metadata:
    """Structure of information of time series."""
    data: jnp.ndarray
    time_array = None
    name: str | None = None
    parameters: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    optimization_info: dict = field(default_factory=dict)