import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax.scipy.stats import norm
from fractrics.helper import partition_map

def pareto_shape(ts, order=None):
    """
    Computes the Pareto Shape index as the inverse of Hill's tail index of a time series.
    """
    ts = jnp.asarray(ts)
    ordst = jnp.sort(ts)
    n = ordst.shape[0]

    if order is None:
        order = jnp.array([int(n * p / 100) for p in [1, 5, 7, 10]])
    order = jnp.atleast_1d(order)

    def tail_index(o):
        indices = jnp.arange(n)
        mask = indices >= o
        tail = jnp.where(mask, ordst, 1.0)  # replace unused entries with 1.0 to avoid log(0)
        ref = ordst[o]
        ratio = jnp.where(mask, tail / ref, 1.0)
        logs = jnp.log(jnp.abs(ratio))
        sum_logs = jnp.sum(jnp.where(mask, logs, 0.0))
        return sum_logs / o

    tail_indices = vmap(tail_index)(order)
    return tail_indices ** (-1)

def tail_pv(ps1, ps2, order=[1, 5, 7 , 10]):
    """
    Computes the p-value/array of p-values of 2 pareto-shape statistics
    """
    order = jnp.array(order)
    if ps1.shape != ps2.shape or ps1.shape != order.shape: raise ValueError("order and Pareto statistics must be of same length")
    nabsZ = -jnp.abs((ps1 - ps2)*jnp.sqrt(order/2))
    return 2 * norm.cdf(nabsZ, loc=0, scale=1)

def var_FR(ts, var, n=1, f_agg=np.sum):
    """
    Ratio of times a series falls below the VaR specified
    :param ts: array of observations
    :param var: Value-at-Risk Scalar
    :param n: length of aggregation
    :param f_agg: aggregation function. Default is sum
    """ 
    agg = partition_map(ts, f_agg,n)
    return np.sum(agg<var)/len(agg)
