# Implementation of bivariate t-copula

import jax.numpy as jnp
import jax.scipy as jsp

from jax import lax
from fractrics.solvers import nelder_mead
from fractrics._components.copula import rank_to_uniform

def t_cdf(df, t):
    x = df / (df + t**2)
    ibeta = jsp.special.betainc(df / 2, 0.5, x)
    return 0.5 + jnp.sign(t) * 0.5 * (1 - ibeta)

def t_ppf(u, df, tol=1e-8, max_iter=60):
    """JIT-compatible inverse CDF (ppf) for Student-t via vectorized binary search."""
    lo = jnp.full_like(u, -20.0)
    hi = jnp.full_like(u,  20.0)

    def cond_fun(state):
        i, lo, hi = state
        return (i < max_iter) & (jnp.max(hi - lo) > tol)

    def body_fun(state):
        i, lo, hi = state
        mid = 0.5 * (lo + hi)
        cdf_mid = t_cdf(df, mid)
        lo = jnp.where(cdf_mid < u, mid, lo)
        hi = jnp.where(cdf_mid >= u, mid, hi)
        return i + 1, lo, hi

    _, lo, hi = lax.while_loop(cond_fun, body_fun, (0, lo, hi))
    return 0.5 * (lo + hi)

def t_copula(x:jnp.ndarray, y:jnp.ndarray, init_guess, max_iter) -> tuple[jnp.ndarray, float, bool, float]:
    """Bivariate T-copula"""
    
    x_uniform = rank_to_uniform(x)
    y_uniform = rank_to_uniform(y)
    Unif = jnp.column_stack([x_uniform, y_uniform])
    
    def constrain(params:jnp.ndarray):
        corr, df = params
        return jnp.array([jnp.tanh(corr), 2.0 + jnp.exp(df)])
    
    def deconstrain(params):
        corr, df = params
        return jnp.array([jnp.arctanh(corr), jnp.log(df - 2.0)])
    
    def nll(params:jnp.ndarray):
        params_constrained = constrain(params)
        corr, df = params_constrained
        
        z =jnp.column_stack([t_ppf(Unif[:,0], df), t_ppf(Unif[:,1], df)])

        R = jnp.array([[1.0, corr],[corr, 1.0]])
        Ri = jnp.linalg.inv(R)
        detR = jnp.linalg.det(R)

        q = jnp.einsum('ti,ij,tj->t', z, Ri, z)
        d = 2
        const = jsp.special.gammaln((df + d)/2) - jsp.special.gammaln(df/2) \
                - 0.5 * (jnp.log(jnp.pi*df) * d + jnp.log(detR))
        log_ft2 = const - 0.5*(df + d)*jnp.log1p(q/df)

        log_ft1 = jsp.stats.t.logpdf(z, df=df)
        return -jnp.sum(log_ft2 - jnp.sum(log_ft1, axis=1))

    init_guess_deco = deconstrain(init_guess)
    
    result = nelder_mead.solver(init_guess_deco, nll, max_iter=max_iter)
    fitted = constrain(result[0])
    return fitted, result[1], result[2], result[3]