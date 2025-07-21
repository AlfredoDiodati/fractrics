# Contains abstract time series classes. 
# Stochastic classes have an additional method containing simulation.


import numpy as np
import pandas as pd

import jax.numpy as jnp
from abc import abstractmethod, ABC

class time_series(ABC):
    """Generic time series class."""
    
    def __init__(self, ts: np.ndarray | jnp.ndarray | pd.Series | pd.DataFrame, name: str | None = None) -> None:
        """
        Initializes the model, data to use, hyperparameters, ...
        ts: time series to initialize the model to.
        """
        self.ts = jnp.array(ts)
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def fit(self) -> jnp.ndarray:
        """Returns the model's parameters using a loss function."""
        pass
    
    @abstractmethod
    def forecast(self, filtration) -> jnp.ndarray:
        """Makes a prediction with the fitted model.
        filtration: horizon of forecast or new independent variables to use
        """
        pass
            
class stochastic_ts(time_series):
    """Time series with random components"""

    @abstractmethod
    def simulation(self, nsim: int)-> jnp.ndarray:
        """Generates nsim datapoints from the model"""
        pass