from typing import NamedTuple
import jax, jax.numpy as jnp
from functools import partial

class UKFState(NamedTuple):
    x: jnp.ndarray  # (dim_x,)
    P: jnp.ndarray  # (dim_x, dim_x)

def sigma_points(x, P, c):
    A = jnp.linalg.cholesky(c * P)
    dev = jnp.concatenate([A, -A], axis=1)   
    return jnp.vstack([x[None,:], (x + dev.T)])

def predict(x, P, fx, fx_args, Q, Wm, Wc, dt, c):
        
    sigmas = sigma_points(x, P, c)
       
    sigmas_f = jax.vmap(lambda s: fx(s, dt, fx_args))(sigmas)
    x_pred = jnp.dot(sigmas_f.T, Wm)
    diff = sigmas_f - x_pred  
    cross = jnp.einsum('bi,bj->bij', diff, diff)
    P_pred = Q + jnp.sum(Wc[:, None, None] * cross, axis=0)

    return x_pred, P_pred

def update(x, P, z, hx, hx_args, R, Wm, Wc, c):
    
    sigmas = sigma_points(x, P, c)
    sigmas_h = jax.vmap(lambda s: hx(s, hx_args))(sigmas)
    z_pred = jnp.dot(Wm, sigmas_h) 
    dz = sigmas_h - z_pred
    
    Pz = R + jnp.einsum('i,ij,ik->jk', Wc, dz, dz)

    # spectral projection for numerical stability
    Pz = 0.5 * (Pz + Pz.T)
    eigvals, eigvecs = jnp.linalg.eigh(Pz)
    min_eig = jnp.min(eigvals)
    eigvals_clipped = jnp.clip(eigvals, a_min=1e-6)
    Pz = (eigvecs * eigvals_clipped) @ eigvecs.T
    
    eps = 1e-6
    Pz = jnp.where(
        min_eig < eps,
        Pz + (eps - min_eig + 1e-8) * jnp.eye(Pz.shape[0]),
        Pz
    )

    dx = sigmas - x
    
    Pxz = jnp.einsum('i,ij,ik->jk', Wc, dx, dz)
    
    K = jnp.linalg.solve(Pz, Pxz.T).T
    
    x_new = x + K @ (z - z_pred)
    
    P_new = P - K @ Pz @ K.T
    P_new = 0.5*(P_new + P_new.T)
    eigvals, eigvecs = jnp.linalg.eigh(P_new)
    eigvals_clipped = jnp.clip(eigvals, a_min=1e-6)
    P_new = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T
    
    err = z - z_pred
    ll = -0.5*(jnp.linalg.slogdet(Pz)[1] + err.T @ jnp.linalg.inv(Pz) @ err + z.size*jnp.log(2*jnp.pi))
    
    return x_new, P_new, ll

@partial(jax.jit, static_argnames=('fx','hx'))
def ukf_step(state: UKFState, z, fx, fx_args, hx, hx_args, Q, R, Wm, Wc, dt, c):
    (x, P) = state

    x_pred, P_pred = predict(x, P, fx, fx_args, Q, Wm, Wc, dt, c)
    x_new, P_new, ll = update(x_pred, P_pred, z, hx, hx_args, R, Wm, Wc, c)
    return UKFState(x_new, P_new), -ll
