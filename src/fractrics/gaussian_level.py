import jax as jnp
from jax import lax
from fractrics import nelder_mead
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass, replace, field
from fractrics._components.core import ts_metadata

@dataclass(frozen=True)
class metadata(ts_metadata):
    data: jnp.ndarray | None = None
    parameters : dict = field(default_factory = lambda: {
        'omega': None,
        'alpha': None,
        'beta': None,
        'sigma': None
    })
    optimization_info : dict = field(default_factory= lambda: {
        'negative_log_likelihood': None,
        'is_converged': None,
        'n_iteration': None
    })
    noise : jnp.ndarray | None = None
    initial_filter: jnp.ndarray | None = None
    filtered: jnp.ndarray | None = None

def filter(model: metadata) -> metadata:
    data = jnp.asarray(model.data)
    omega = jnp.asarray(model.parameters['omega'])
    alpha = jnp.asarray(model.parameters['alpha'])
    beta  = jnp.asarray(model.parameters['beta'])
    init_f = jnp.asarray(model.initial_filter)

    last_index = data.shape[0] - 1
    realized_rate_tm1 = data[:-1]
    time_indexes = jnp.arange(1, last_index + 1)

    def case_beta_one(_):
        return init_f + time_indexes * omega + alpha * jnp.cumsum(realized_rate_tm1)

    def case_beta_zero(_):
        return omega + alpha * realized_rate_tm1

    def case_general(_):
        beta_powers = beta ** time_indexes
        vectorized_component = beta_powers * init_f + omega * (1.0 - beta_powers) / (1.0 - beta)

        def scan_body(carry, x):
            new = beta * carry + x
            return new, new

        _, recursive_component = lax.scan(scan_body, jnp.asarray(0.0, dtype=data.dtype), realized_rate_tm1)
        return vectorized_component + alpha * recursive_component

    mu_body = lax.cond(
        jnp.isclose(beta, 1.0),
        case_beta_one,
        lambda _: lax.cond(jnp.isclose(beta, 0.0), case_beta_zero, case_general, None),
        None
    )

    mu_array_tail = mu_body(None)
    mu_array = jnp.concatenate([jnp.reshape(init_f, (1,)), mu_array_tail])

    return mu_array

def fit(model: metadata) -> metadata:
    
    initial_guess, unravel_f = ravel_pytree(model.parameters)
    data = model.data
    
    def nll(params):
        par_dict = unravel_f(params)
        new_model = replace(model, parameters=par_dict)
        filtered = filter(new_model)
        sigma_squared = par_dict['sigma']**2
        constant_term = -jnp.log(jnp.sqrt(2*jnp.pi*sigma_squared)) * data.shape[0]
        return -(constant_term + jnp.sum(-((data-filtered)**2 / (2*sigma_squared))))
    
    params, nll_v, is_converged, num_iteration = nelder_mead.solver(initial_guess, nll)
    
    return replace(model,
        parameters = unravel_f(params),
        optimization_info = {
        'negative_log_likelihood': nll_v,
        'is_converged': is_converged,
        'n_iteration': num_iteration
        }
    )