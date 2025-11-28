# risk related functions

import jax.numpy as jnp
import jax.scipy.stats as jss
from jax import vmap

from fractrics.solvers import nelder_mead

def hill(ts: jnp.ndarray, order: jnp.ndarray | None = None) -> jnp.ndarray:
    """ Hill estimator using upper order statistics.
    
    Args:
        ts: 1D array of positive samples.
        order: Array of integers (number of top order statistics to use).
               If None, defaults to [1%, 5%, 7%, 10%] of the sample size.

    Returns:
        1D array of estimated Hill shape parameters for each order.
    """
    ts = jnp.asarray(ts)
    ordst = jnp.sort(ts)[::-1]
    n = ordst.shape[0]

    if order is None:
        order = jnp.array([int(n * p / 100) for p in (1, 5, 7, 10)], dtype=jnp.int32)
    order = jnp.clip(order, 1, n - 1)
    order = jnp.atleast_1d(order)

    idx = jnp.arange(n)

    def hill_for_k(k):
        mask = idx < k
        x_top = jnp.where(mask, ordst, 1.0)
        x_ref = ordst[k - 1]
        logs = jnp.log(x_top / x_ref) * mask
        return jnp.sum(logs) / k

    hills = vmap(hill_for_k)(order)
    return hills

def pareto_shape(ts: jnp.ndarray, order: jnp.ndarray | None = None) -> jnp.ndarray:
    """Inverse of Hill's estimator"""
    return 1.0/hill(ts, order)

def tail_pv(ps1, ps2, order=[1, 5, 7 , 10]):
    """
    Computes the p-value/array of p-values of 2 pareto-shape statistics
    """
    order = jnp.array(order)
    if ps1.shape != ps2.shape or ps1.shape != order.shape: raise ValueError("order and Pareto statistics must be of same length")
    
    nabsZ = -jnp.abs((ps1 - ps2)*jnp.sqrt(order/2))
    return 2 * jss.norm.cdf(nabsZ, loc=0, scale=1)

def gpareto_tail(ts:jnp.ndarray, initial_guess: jnp.ndarray, location = 0.0) -> tuple[jnp.ndarray, float, bool, float]:
    """Fit a generalized pareto distribution to data. Guess: [scale, shape]"""
    
    def nll(guess:jnp.ndarray) -> float:
        sigma, xi = guess
        sigma_positive = jnp.exp(sigma)
        z = 1 + xi * ( ts- location) / sigma_positive
        return -jnp.sum(-jnp.log(sigma_positive) - (1/xi + 1) * jnp.log(z))
    
    def deconstrain(guess:jnp.ndarray)->jnp.ndarray:
        sigma, xi = guess
        return jnp.array([jnp.log(sigma), xi])

    deconstrained_initial_guess = deconstrain(initial_guess)
    
    result = nelder_mead.solver(deconstrained_initial_guess, nll)
    
    fitted = jnp.exp(result[0])
    return fitted, result[1], result[2], result[3]

def lower_tail_dependence(x:jnp.ndarray, y:jnp.ndarray, alpha:float|None = None, k:int|None = None):
    x_sorted = jnp.sort(x, descending=False)
    y_sorted = jnp.sort(y, descending=False)
    k = int(x.shape[0] * alpha) if k is None else k
    x_k = x_sorted[k - 1]
    y_k = y_sorted[k - 1]
    is_x_ltk = x <= x_k
    is_y_ltk = y <= y_k
    
    is_joint_ltk = is_x_ltk * is_y_ltk
    return jnp.sum(is_joint_ltk) / k

def upper_tail_dependence(x:jnp.ndarray, y:jnp.ndarray, alpha:float|None = None, k:int|None = None):
    x_sorted = jnp.sort(x, descending=False)
    y_sorted = jnp.sort(y, descending=False)
    k = int(x.shape[0] * alpha) if k is None else k
    x_nmk = x_sorted[x.shape - k - 1]
    y_nmk = y_sorted[x.shape - k - 1]
    is_x_ltk = x > x_nmk
    is_y_ltk = y > y_nmk
    
    is_joint_ltk = is_x_ltk * is_y_ltk
    return jnp.sum(is_joint_ltk) / k