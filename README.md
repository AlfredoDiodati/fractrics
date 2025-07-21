# fractrics

[![PyPI - Version](https://img.shields.io/pypi/v/fractrics.svg)](https://pypi.org/project/fractrics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fractrics.svg)](https://pypi.org/project/fractrics)

-----

## Table of Contents

- [Installation](#installation)
- [Quick example](#quick-example)
- [License](#license)
- [Planned updates](#planned-updates)
- [References](#references)

## Installation

```console
pip install fractrics
```

## Quick example

The main tool in fractrics is the MSM class, an implementation of the univariate [Markov Switching Multifractal Model](https://en.wikipedia.org/wiki/Markov_switching_multifractal). The logaritmic difference between observations is modeled as the noise-adjusted square root of the product of a chosen number of latent volatility components, each following the dynamics of discrete first order markov chains, whose transition depends on geometrically-spaced Poisson arrivals, and an unconditional term, effectively being the unconditional volatility.

Such structure effectively captures the behaviour of time series with fat tails, hyperbolic correlation decay, and multifractal moments, such as the returns of many financial assets.

The implementation is made in JAX, simplifying parallelization of the code. Moreover, following from [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5276090) paper, the memory complexity of the forward algorithm is reduced, due to the factorization of latent states.

To use the model, start with an example time series. Note that the model is only defined for positive time series (as it was created to model prices of financial assets).


```python
from fractrics.time_series.MSM import MSM
import jax.numpy as jnp
import numpy as np

ts_test = jnp.array(np.random.normal(50, 10, 10))
```

Then initialize the model. It requires the following hyperparameters:
 - `num_latent`: how many volatility components, integer.
 - `marg_prob_mass`: the probability mass of the marginal distribution of the latent states, needs to sum to 1. 


```python
model = MSM(ts=ts_test, num_latent=3)
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
fitresult = model.fit(initial_parameters=initial_params, maxiter=1000, verbose=False)

params, current_distribution, transition_tensor, distribution_history, negative_log_likelihood = fitresult
print(params)

```

    [0.22855803 3.1763792  0.44004682 1.0000163  0.9999837 ]


It is also possible to make simulations with the MSM. To avoid tracing issues with JAX, the parameters are to be given as input for the simulation. Follows an example with the parameters of the fitted model above.


```python
num_latent = model.num_latent
unconditional_term = params[0]
arrival_gdistance = params[1]
hf_arrival = params[2]
marg_support = params[3:]

poisson_arrivals = 1 - (1 - hf_arrival) ** (1 / (arrival_gdistance ** (jnp.arange(num_latent, 0, -1) - 1)))

simulation = model.simulation(number_simulations= 1000, unconditional_term = unconditional_term,
                              poisson_arrivals=poisson_arrivals,
                              marginal_support = marg_support
)

```

Finally a 7 period forecast. The transition tensor and current distribution are required as input


```python
forecast = model.forecast(7, current_distribution, *transition_tensor)
```


## License

`fractrics` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Planned updates

- refactoring the functions in `_pending_refactor`.
- `diagnostics.py`: adding other common metrics.
- `_ts_components/_HMM/base.py`:
    - implementing viterbi and backwards algorithms
    - generalize components of the forward algorithms that apply to other hidden markov models
- `MSM`:
    - use pytrees or other forms of custom class to facilitate the usage of the API and store valuable information of fitted model.
    - implement standard errors and robust standard erros of the parameters (pseudo-code commented in already)
    - implement model selection metrics
    - model implied moments, value at risk.
    - Allow for creating simulations without initializing the model with a time series.
- `level_MSM` re-implement the model using the new architecture of the package

## References

- Calvet, L.E. and Fisher, A.J. (2004). How to Forecast Long-Run Volatility: Regime Switching and the Estimation of Multifractal Processes. Journal of Financial Econometrics, 2(1).

- Calvet, L.E. and Fisher, A.J. (2008). Multifractal Volatility. Theory, Forecasting, and Pricing. Academic Press.

- Calvet, L.E., Fisher, A.J. and Thompson, S.B. (2004). Volatility Comovement: A Multifrequency Approach. SSRN Electronic Journal. doi:https://doi.org/10.2139/ssrn.582541.

- Diodati, A. (2025). Tensor representation of Markov Switching Multifractal Models. doi:https://doi.org/10.2139/ssrn.5276090.

- Lux, T. (2008). The Markov-Switching Multifractal Model of Asset Returns. Journal of Business & Economic Statistics, 26(2), pp.194–210. doi:https://doi.org/10.1198/073500107000000403.

- Lux, T. (2020). Inference for Nonlinear State Space Models: A Comparison of Different Methods applied to Markov-Switching Multifractal Models. Econometrics and Statistics. doi:https://doi.org/10.1016/j.ecosta.2020.03.001.

- Lux, T., Morales-Arias, L. and Sattarhoff, C. (2011). A Markov-switching multifractal approach to forecasting realized volatility. [online] Kiel Working Papers. Available at: https://ideas.repec.org/p/zbw/ifwkwp/1737.html [Accessed 30 May 2025].

- Murphy, K.P. (2012). Machine learning : a probabilistic perspective. Cambridge (Ma): Mit Press.

- Rypdal, M. and Løvsletten, O. (2011). Multifractal modeling of short-term interest rates. arXiv (Cornell University).