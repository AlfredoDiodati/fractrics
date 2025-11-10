import jax.numpy as jnp
from jax import vmap, lax, jit
 
def _vmap_eval(f: callable, pts: jnp.ndarray) -> jnp.ndarray:
    """f should accept (n,) and return a scalar jnp.array()"""
    return vmap(f)(pts)

def solver(
    initial_guess: jnp.ndarray,
    f: callable,
    max_iter: int = 1000,
    tol: float = 1e-6,
    reflection_factor: float = 1.0,
    expansion_factor: float = 2.0,
    contraction_factor: float = 0.5,
    shrink_factor: float = 0.5) -> tuple[jnp.ndarray, float, bool, float]:
    """f must be JAX-traceable and return a scalar jnp.ndarray.
        returns: solution, score value, is_converged (boolean), number of iterations
    """
    n = initial_guess.shape[0]

    scale = 0.05 * jnp.maximum(jnp.abs(initial_guess), 1.0)
    perturb = initial_guess[None, :] + (scale[:, None] * jnp.eye(n))
    simplex = jnp.vstack([initial_guess[None, :], perturb])

    fvals = _vmap_eval(f, simplex)
    state = (simplex, fvals, jnp.array(0), jnp.array(False))

    def cond_fun(state):
        _, _, it, done = state
        not_done = jnp.logical_not(done)
        return jnp.logical_and(it < max_iter, not_done)

    def body_fun(state):
        simplex, fvals, it, is_converged = state

        order = jnp.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        best = simplex[0]
        f_best = fvals[0]
        f_second = fvals[-2]
        worst = simplex[-1]
        f_worst = fvals[-1]

        spread = jnp.max(fvals) - jnp.min(fvals)
        is_converged = spread <= tol

        def after_converged(_):
            return simplex, fvals, it + 1, jnp.array(True)

        def not_converged(_):
            centroid_noworst = jnp.mean(simplex[:-1], axis=0)
            reflection = centroid_noworst + reflection_factor * (centroid_noworst - worst)
            f_reflection = f(reflection)

            def accept_reflection(_):
                """reflection is between best and second-best"""
                new_simplex = simplex.at[-1].set(reflection)
                new_fvals = fvals.at[-1].set(f_reflection)
                return new_simplex, new_fvals

            def try_expansion(_):
                """reflection better than best"""
                expansion = centroid_noworst + expansion_factor * (reflection - centroid_noworst)
                f_expansion = f(expansion)
                def choose_exp(_):
                    new_simplex = simplex.at[-1].set(expansion)
                    new_fvals = fvals.at[-1].set(f_expansion)
                    return new_simplex, new_fvals
                def choose_ref(_):
                    new_simplex = simplex.at[-1].set(reflection)
                    new_fvals = fvals.at[-1].set(f_reflection)
                    return new_simplex, new_fvals
                take_exp = f_expansion < f_reflection
                return lax.cond(take_exp, choose_exp, choose_ref, operand=None)

            def outside_contraction(_):
                """reflection is better than worst but not better than second"""
                contraction = centroid_noworst + contraction_factor * (reflection - centroid_noworst)
                f_contraction = f(contraction)
                def accept_contract(_):
                    new_simplex = simplex.at[-1].set(contraction)
                    new_fvals = fvals.at[-1].set(f_contraction)
                    return new_simplex, new_fvals
                def shrink_around_best(_):
                    shrunk = best + shrink_factor * (simplex - best)
                    new_fvals = _vmap_eval(f, shrunk)
                    return shrunk, new_fvals
                accept = f_contraction <= f_reflection
                return lax.cond(accept, accept_contract, shrink_around_best, operand=None)

            def inside_contraction(_):
                """reflection >= worst"""
                contraction = centroid_noworst + contraction_factor * (worst - centroid_noworst)
                f_contraction = f(contraction)
                def accept_contract(_):
                    new_simplex = simplex.at[-1].set(contraction)
                    new_fvals = fvals.at[-1].set(f_contraction)
                    return new_simplex, new_fvals
                def do_shrink(_):
                    shrunk = best + shrink_factor * (simplex - best)
                    new_fvals = _vmap_eval(f, shrunk)
                    return shrunk, new_fvals
                accept = f_contraction < f_worst
                return lax.cond(accept, accept_contract, do_shrink, operand=None)

            b1 = jnp.logical_and(f_reflection >= f_best, f_reflection < f_second)
            b2 = f_reflection < f_best
            b3a = jnp.logical_and(f_reflection >= f_second, f_reflection < f_worst)

            def branch1(_): return accept_reflection(None)
            def branch2(_): return try_expansion(None)
            def branch3a(_): return outside_contraction(None)
            def branch3b(_): return inside_contraction(None)

            def arb_choice(_):
                res = lax.cond(b1, branch1,
                        lambda _: lax.cond(b2, branch2,
                            lambda _: lax.cond(b3a, branch3a, branch3b, operand=None),
                        operand=None),
                    operand=None)
                return res

            new_simplex, new_fvals = arb_choice(None)
            return new_simplex, new_fvals, it + 1, jnp.array(False)

        return lax.cond(is_converged, after_converged, not_converged, operand=None)

    simplex, fvals, num_iteration, is_converged = lax.while_loop(cond_fun, body_fun, state)
    order = jnp.argsort(fvals)
    simplex = simplex[order]
    fvals = fvals[order]
    return simplex[0], fvals[0], is_converged, num_iteration

solver = jit(solver, static_argnames=('f',))

if __name__ == "__main__" and __debug__:
    def quad(x):
        x = jnp.asarray(x)
        return jnp.sum(x ** 2)

    x0 = jnp.array([3564.0])
    best_point, best_val, is_converged, num_iteration = solver(x0, quad, max_iter=2000, tol=0.0)
    print("best_point:", best_point, "best_val:", best_val, "is_converged:", is_converged, "n. iterations", num_iteration)