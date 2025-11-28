import jax.numpy as jnp
from fractrics import nelder_mead
from fractrics._components.copula import rank_to_uniform

def clayton_logpdf(u, v, theta, eps=1e-12):
    # clip to avoid log(0) and powers at 0/1
    u = jnp.clip(u, eps, 1.0 - eps)
    v = jnp.clip(v, eps, 1.0 - eps)

    a = u**(-theta) + v**(-theta) - 1.0
    return (
        jnp.log1p(theta)
        - (1.0 + theta) * (jnp.log(u) + jnp.log(v))
        - (2.0 + 1.0/theta) * jnp.log(a)
    )

def clayton_copula(x: jnp.ndarray,
                       y: jnp.ndarray,
                       init_theta: float = 1.0,
                       max_iter: int = 200):
    """
    Returns (theta_hat, negloglik, success, n_iter)
    """

    u = rank_to_uniform(x)
    v = rank_to_uniform(y)

    def constrain(eta):
        return jnp.exp(eta)

    def deconstrain(theta):
        return jnp.log(theta)

    def nll(eta):
        theta = constrain(eta)
        ll = clayton_logpdf(u, v, theta)
        return -jnp.sum(ll)

    eta0 = deconstrain(jnp.asarray(init_theta))
    eta_hat, fval, success, n_iter = nelder_mead.solver(jnp.array([eta0]), nll, max_iter=max_iter)

    theta_hat = constrain(eta_hat)
    return theta_hat, fval, success, n_iter

def clayton_kendall_tau(theta):
    return theta / (theta + 2.0)

def clayton_lower_tail_dependence(theta):
    return 2.0 ** (-1.0 / theta)
