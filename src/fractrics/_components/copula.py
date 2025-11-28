import jax.numpy as jnp

def rank_to_uniform(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform returns to uniforms via ranks.
    Equivalent to scipy.stats.rankdata(method='average') / (T+1).
    """
    T = x.size
    sorted_x = jnp.sort(x)

    def avg_rank(val):
        idx = jnp.searchsorted(sorted_x, val, side='left')
        cnt = jnp.searchsorted(sorted_x, val, side='right') - idx
        return (idx + 1 + (idx + cnt)) / 2

    avg_rank_vmap = jnp.vectorize(avg_rank)
    ranks = avg_rank_vmap(x)
    return ranks / (T + 1)