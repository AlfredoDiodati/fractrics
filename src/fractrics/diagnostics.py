"""Statistics to test performances of models"""

import jax.numpy as jnp
import jax.scipy.stats as jss

def pareto_shape(ts: jnp.ndarray, order: jnp.ndarray | None) -> jnp.ndarray:
    """Computes the Pareto Shape index as the inverse of Hill's tail index of a time series."""
    ts = jnp.asarray(ts)
    ordst = jnp.sort(ts)
    n = ordst.shape[0]

    if order is None:
        order = jnp.array([n * p // 100 for p in (1, 5, 7, 10)], dtype=jnp.int32)
    order = jnp.atleast_1d(order).astype(jnp.int32)

    idx = jnp.arange(n)
    mask = idx[None, :] >= order[:, None]

    ref = ordst[order]
    ref = ref[:, None]

    tail = jnp.where(mask, ordst[None, :], 1.0)
    logs = jnp.log(tail / ref)

    hill = jnp.sum(logs, axis=1) / order
    return 1.0 / hill

def tail_pv(ps1, ps2, order=[1, 5, 7 , 10]):
    """
    Computes the p-value/array of p-values of 2 pareto-shape statistics
    """
    order = jnp.array(order)
    if ps1.shape != ps2.shape or ps1.shape != order.shape: raise ValueError("order and Pareto statistics must be of same length")
    
    nabsZ = -jnp.abs((ps1 - ps2)*jnp.sqrt(order/2))
    return 2 * jss.norm.cdf(nabsZ, loc=0, scale=1)