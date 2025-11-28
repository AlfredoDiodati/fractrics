import jax.numpy as jnp
from jax import grad, lax, jit

def solver(
    initial_guess: jnp.ndarray,
    f: callable,
    max_iter: int = 1000,
    tol: float = 1e-6,
    lr: float = 1e-2,
    momentum: float = 0.9,
) -> tuple[jnp.ndarray, float, bool, float]:
    """
    f must be JAX-traceable and return scalar jnp.ndarray.
    Returns: solution, score value, is_converged, num_iterations.
    """

    g = grad(f)
    theta0 = initial_guess
    v0 = jnp.zeros_like(initial_guess)

    state = (theta0, v0, jnp.array(0), jnp.array(False))

    def cond_fun(state):
        theta, v, it, done = state
        not_done = jnp.logical_not(done)
        return jnp.logical_and(it < max_iter, not_done)

    def body_fun(state):
        theta, v, it, done = state

        grad_theta = g(theta)
        v_new = momentum * v + grad_theta
        theta_new = theta - lr * v_new

        is_converged = jnp.linalg.norm(grad_theta) < tol

        return theta_new, v_new, it + 1, is_converged

    theta_final, v_final, num_iter, is_converged = lax.while_loop(
        cond_fun, body_fun, state
    )

    return theta_final, f(theta_final), is_converged, num_iter

solver = jit(solver, static_argnames=("f",))

if __name__ == "__main__" and __debug__:
    def quad(x):
        x = jnp.asarray(x)
        return jnp.sum(x**2)

    x0 = jnp.array([3564.0])
    sol, val, conv, it = solver(x0, quad, max_iter=5000, lr=0.01, tol=1e-12)
    print("sol:", sol, "val:", val, "converged:", conv, "iters:", it)
