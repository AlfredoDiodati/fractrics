# fractrics

[![PyPI - Version](https://img.shields.io/pypi/v/fractrics.svg)](https://pypi.org/project/fractrics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fractrics.svg)](https://pypi.org/project/fractrics)

-----

## Table of Contents

- [Installation](#installation)
- [Quick example](#quick-example)
- [Project Structure](#project-structure)
- [Planned updates](#planned-updates)
- [References](#references)
- [License](#license)

## Installation

```console
pip install fractrics
```

## Quick example

The main tool in fractrics is the MSM class, an implementation of the univariate [Markov Switching Multifractal Model](https://en.wikipedia.org/wiki/Markov_switching_multifractal). The logaritmic difference between observations is modeled as the noise-adjusted square root of the product of a chosen number of latent volatility components, each following the dynamics of discrete first order markov chains, whose transition depends on geometrically-spaced Poisson arrivals, and an unconditional term, effectively being the unconditional volatility.

Such structure effectively captures the behaviour of time series with fat tails, hyperbolic correlation decay, and multifractal moments, such as the returns of many financial assets.

The implementation is made in JAX, simplifying parallelization of the code. Moreover, following from [this](https://link.springer.com/article/10.1023/A:1007425814087) paper, the memory complexity of the forward algorithm is reduced, due to the factorization of latent states.

To use the model, start with an example time series. Note that the model is only defined for positive time series (as it was created to model prices of financial assets).


```python
from fractrics.time_series.MSM import MSM
from fractrics.utilities import summary
import jax.numpy as jnp
import numpy as np

ts_test = np.loadtxt("data/msm_simulation.csv")[:50]
```

Then initialize the model. It requires the following hyperparameters:
 - `n_latent`: how many volatility components, integer.
 - `marg_prob_mass`: the probability mass of the marginal distribution of the latent states, needs to sum to 1. 


```python
model = MSM(ts=ts_test, n_latent=3)
```

To fit the model to the data, start with an initial guess. The `MSM.fit()` method then optimizes the parameters using `jaxopt`'s [Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).

By assumption, all the parameters need to be positive, and have further individual constrains:

- `marg_support`: the support of the marginal probability mass defined in the parameters. It needs to have unity expectation. In the symmetric binomial case, this can be enforced by specifying one value $m_0$, and having the second value be $2 - m_0$.

- `unconditional_term`: the unconditional distribution of the model, a positive double.

- `arrival_gdistance`: the geometric distance between the Poisson arrivals of each latent volatility component, a positive double.

- `hf_arrival`: the highest poisson arrival probability (i.e. the proability of state switch of the highest frequency component).

Note: to maintain the constrains during optimization, the parameters are transformed using mappings.


```python
initial_params = jnp.array([
    2,    #unconditional term
    3.0,    #arrival_gdistance
    0.98,   #hf_arrival
    #support
    1.5,    
    0.5
])

msm_result = model.fit(initial_parameters=initial_params, maxiter=1000)
```

`msm_result` is a custom dataclass (`msm_metadata`) that contains relevant information about the model. This construct reduces the verbosity of the API, as it can be passed as the only input required to operate with the following methods.

It contains:
- `filtered`: a dictionary containing the current distribution of the latent components, the list of distribution list at each time step, inferred using the forward algorithm, the transition tensor of the model (in factor form), and the vector of latent states
- `parameters`: a dictionary containing the model parameters
- `standard_errors`: a dictionary containing the model standard errors
- `robust_standard_errors`: a dictionary containing the [Eicker–Huber–White](https://en.wikipedia.org/wiki/Heteroskedasticity-consistent_standard_errors) standard errors
- `hyperparameters:` a dictionary containing the hyperparameters of the model (the number of volatility components and the marginal probability mass)
- `optimization_info`: information about the optimization process
- `name`: the internal name of the model (defaults to "MSM")
- `data`: the input data
- `data_log_change`: the logarithmic change between each data point and its next observation (e.g. the log. return if the original data is a series of financial prices).

Most of this information can be printed using the `summary()` function. Note: if the attribute `latex` is True, then summary will print a latex table.


```python
summary(msm_result)
```

    ------------------  -------------------------  -------------------
    model:              MSM
    ------------------  -------------------------  -------------------
    Hyperparameters
    ------------------  -------------------------  -------------------
    n_latent            marginal_probability_mass
    3                   [0.5 0.5]
    ------------------  -------------------------  -------------------
                        Parameters                 Standard Errors
    ------------------  -------------------------  -------------------
    unconditional_term  0.6317223310470581         0.06381293386220932
    arrival_gdistance   3.0993151664733887         nan
    hf_arrival          0.5677862763404846         nan
    marginal_support    [1.        1.0000001]      [0.    0.125]
    ------------------  -------------------------  -------------------
    Likelihood:         -20.905973434448242
    ------------------  -------------------------  -------------------


It is also possible to make simulations with the MSM. The `MSM.simulation` method takes a `msm_metadata` object as input to choose the parameters, as it is intended to be used to simulate data from a fitted model, as above. If the user wants to simulate from chosen parameters, a `msm_metadata` object needs to be initialized with them.

Follows an example with the parameters of the fitted model above. It returns a tuple containing the simulated logarithmic change (e.g. 1 step return) and corresponding implied volatility.


```python
ret, vol = model.simulation(n_simulations = 1000, model_info = msm_result, seed=123)
```

Finally a 7 period forecast. The method returns the predictive distribution at each forecast horizon, so that it may be used for both point-expectation and uncertainty intervals.


```python
forecast = model.forecast(horizon=7, model_info=msm_result)
```


## Project Structure
```
.
├── notebooks                     # [example jupyter notebooks]
└── src/fractrics                 # [main code repository]
    ├── _pending_refactor/        # legacy code that needs to be restructured
    ├── _ts_components/           # abstract classes and methods for time series
    ├── time_series/              # concretization classes for time series models
    ├── utilities.py              # contains summary function
    └── diagnostics.py            # Statistics to test performances of models

```
## Planned updates

- `_ts_components/_HMM/base.py`:
    - implementing viterbi and backwards algorithms
    - generalize components of the forward algorithms that apply to other hidden markov models
- `MSM`:
    - create plot functions.
        - visualize states
        - visualize learning path
    - implement model selection metrics
    - model implied moments, value at risk.
    - Allow for creating simulations without initializing the model with a time series.
- `diagnostics.py`: adding other common metrics.
- refactoring the functions in `_pending_refactor`.

## References

- Calvet, L.E. and Fisher, A.J. (2004). How to Forecast Long-Run Volatility: Regime Switching and the Estimation of Multifractal Processes. Journal of Financial Econometrics, 2(1).

- Calvet, L.E. and Fisher, A.J. (2008). Multifractal Volatility. Theory, Forecasting, and Pricing. Academic Press.

- Calvet, L.E., Fisher, A.J. and Thompson, S.B. (2004). Volatility Comovement: A Multifrequency Approach. SSRN Electronic Journal. doi:https://doi.org/10.2139/ssrn.582541.

- Ghahramani, Z. and Jordan, M.I. (1997). Factorial Hidden Markov Models. Machine Learning, 29(2/3), pp.245–273. doi:https://doi.org/10.1023/a:1007425814087.

- Lux, T. (2008). The Markov-Switching Multifractal Model of Asset Returns. Journal of Business & Economic Statistics, 26(2), pp.194–210. doi:https://doi.org/10.1198/073500107000000403.

- Lux, T. (2020). Inference for Nonlinear State Space Models: A Comparison of Different Methods applied to Markov-Switching Multifractal Models. Econometrics and Statistics. doi:https://doi.org/10.1016/j.ecosta.2020.03.001.

- Lux, T., Morales-Arias, L. and Sattarhoff, C. (2011). A Markov-switching multifractal approach to forecasting realized volatility. [online] Kiel Working Papers. Available at: https://ideas.repec.org/p/zbw/ifwkwp/1737.html [Accessed 30 May 2025].

- Murphy, K.P. (2012). Machine learning : a probabilistic perspective. Cambridge (Ma): Mit Press.

- Rypdal, M. and Løvsletten, O. (2011). Multifractal modeling of short-term interest rates. arXiv (Cornell University).

## License

`fractrics` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.