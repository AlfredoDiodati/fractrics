from jax import numpy as jnp
from jax.scipy.stats import norm, chi2

def z_score(array:jnp.ndarray):
    """Computes the Z-score for an input array"""
    return (array - jnp.nanmean(array))/jnp.nanstd(array)

def kurtosis(array):
    """Computes kurtosis of input array"""
    return jnp.mean((z_score(array)**4))

def skewness(array):
    """Computes skewness of input array"""
    return jnp.mean(z_score(array)**3)

def acf(ts:jnp.ndarray)-> dict:
    """Compute autocorrelation, T-ratio and Ljung-Box and their p-values for all lags of a time series."""
    
    n = len(ts)
    lags = range(n)
    acf = (jnp.correlate(ts - jnp.nanmean(ts), ts - jnp.nanmean(ts), mode='full'))[n-1:] / (jnp.nanvar(ts)*n)
    t_ratio = acf/(jnp.sqrt(1+2*jnp.cumsum(acf**2)))
    pv_t = 2 * (1 -norm.cdf(abs(t_ratio)))
    q_st = n*(n+2)*jnp.cumsum(acf**2/jnp.arange(n, 0, -1))
    pv_q = 1 - chi2.cdf(q_st, lags)
        
    acf_df = {
    'lag': lags,
    'acf': acf,
    't_ratio': t_ratio,
    'pv_t' : pv_t,
    'q_stat': q_st,
    'pv_q': pv_q
    }
    
    return acf_df