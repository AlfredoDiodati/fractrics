from jax import lax
import jax.numpy as jnp
import jax.scipy.stats as jss
from fractrics.solvers import nelder_mead
from dataclasses import dataclass, replace, field
from fractrics._components.core import ts_metadata

#TODO: remove non-jax conditionals, simulation test

@dataclass(frozen=True)
class metadata(ts_metadata):
    """
    Dataclass containing all relevant information about a specific model.
    It is used by the garchml() class to store results of maximizations and similar.
    
    This way, to use the methods of the class one can just input a garchml_result without manually
    inputing all the parameters and data.
    """
    data: jnp.ndarray | None = None
    parameters: dict = field(default_factory=lambda: {
        'alpha': None, 'beta': None, 'gamma': None,
        'delta': None, 'mu': None, 'lbda': None,
        'omega': None, 'nu': None
    })
    optimization_info: dict = field(default_factory=lambda: {
        'log_likelihood': None, 'is_converged': None, 'num_iteration': None
    })
    
    sigma2: jnp.ndarray | None = None
    epsilon: jnp.ndarray | None = None

def filter(garchml_input: metadata, sigma2_1: float) -> metadata:

    data = jnp.asarray(garchml_input.data)
    sigma2_init = jnp.asarray(sigma2_1, dtype=data.dtype)

    alpha = garchml_input.parameters["alpha"]
    beta  = garchml_input.parameters["beta"]
    gamma = garchml_input.parameters["gamma"]
    delta = garchml_input.parameters["delta"]
    mu    = garchml_input.parameters["mu"]
    lbda  = garchml_input.parameters["lbda"]
    omega = garchml_input.parameters["omega"]

    x_tm1 = data[:-1]
    nonrec = alpha + delta * jnp.tanh(-gamma * x_tm1)

    def step(prev_sigma2, x):
        x_prev, nonrec_t = x
        num = x_prev - mu
        term_sq = (num**2) / prev_sigma2 - 2.0 * lbda * num + (lbda**2) * prev_sigma2
        sigma2 = omega + nonrec_t * term_sq + beta * prev_sigma2
        return sigma2, sigma2

    _, sigma2_tail = lax.scan(step, sigma2_init, (x_tm1, nonrec))

    sigma2_array = jnp.concatenate([jnp.reshape(sigma2_init, (1,)), sigma2_tail])

    sigma_array = jnp.sqrt(sigma2_array)
    epsilon_implied = (data - mu - lbda * sigma2_array) / sigma_array

    return replace(garchml_input, sigma2=sigma2_array, epsilon=epsilon_implied)

def impact_news(garchml_input:metadata, shock:jnp.ndarray) -> jnp.ndarray:
    
    alpha = garchml_input.parameters["alpha"]
    gamma = garchml_input.parameters["gamma"]
    delta = garchml_input.parameters["delta"]
    mu = garchml_input.parameters["mu"]
    lbda = garchml_input.parameters["lbda"]
    news_impact = (alpha + delta * jnp.tanh(-gamma*(mu+lbda+shock)))*(shock**2)
    
    return news_impact

def log_likelihood(garchml_input:metadata, sigma2_1:float) -> float:
    nu = garchml_input.parameters["nu"]
    garchml_filtered = filter(garchml_input, sigma2_1)
    return jnp.sum(jss.t.logpdf(garchml_filtered.epsilon, df=nu) - 0.5*jnp.log(garchml_filtered.sigma2))

def fit(garchml_input:metadata, sigma2_1:float, model:str | None = None)->metadata:
    is_garch = (model == "garch")
    is_garchm = (model == "garchm")
            
    def _nll(parameters:tuple[float, ...], sigma2_1:float) -> float:
        if is_garch: a, b, m, o, n = parameters
        elif is_garchm: a, b, m, l, o, n = parameters
        else: 
            a, b, g, d, m, l, o, n = parameters
            if g < 0 or a <= jnp.abs(d): return 1e12
        if b < 0.0 : return 1e12
        
        garchml_fit = replace(garchml_input, parameters={
            'alpha': a, 'beta': b,
            'gamma': 0.0 if is_garch or is_garchm else g,
            'delta': 0.0 if is_garch or is_garchm else d,
            'mu': m, 'lbda': 0.0 if is_garch else l,
            'omega': o,'nu': n
        })
        garchml_filtered = filter(garchml_fit, sigma2_1)
        if not jnp.any(jnp.isfinite(garchml_filtered.sigma2)) or jnp.any(garchml_filtered.sigma2 <= 0):
            return 1e12

        else: return -log_likelihood(garchml_fit, sigma2_1)
    
    a = garchml_input.parameters["alpha"]
    b = garchml_input.parameters["beta"]
    m = garchml_input.parameters["mu"]
    o = garchml_input.parameters["omega"]
    n = garchml_input.parameters["nu"]
    
    if is_garch: params = (a, b, m, o, n)
        
    elif is_garchm: 
        l = garchml_input.parameters["lbda"]
        params = (a, b, m, l, o, n)
        
    else:
        d = garchml_input.parameters["delta"]
        g = garchml_input.parameters["gamma"]
        l = garchml_input.parameters["lbda"]
        params = (a, b, g, d, m, l, o, n)
    
    params, nll_f, is_converged, num_iterations = nelder_mead.solver(params, _nll)
    
    if is_garch: a, b, m, o, n = params
    elif is_garchm: a, b, m, l, o, n = params
    else: a, b, g, d, m, l, o, n = params
    
    garchml_fit = replace(garchml_input, parameters={
            'alpha': a, 'beta': b,
            'gamma': 0.0 if is_garch or is_garchm else g,
            'delta': 0.0 if is_garch or is_garchm else d,
            'mu': m, 'lbda': 0.0 if is_garch else l,
            'omega': o, 'nu': n
        })
    garchml_fit = replace(garchml_fit, 
        optimization_info = {
            'negative_log_likelihood': nll_f,
            'is_converged' : is_converged,
            'num_iterations': num_iterations
        })
    
    return garchml_fit