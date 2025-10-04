import jax
import jax.numpy as jnp
from fractrics.unscent_KF import *

class CascadeDTSM:
    """Cascade dynamic term structure model as in Calvet, Fisher, and Wu (2018)."""
    def __init__(self, rates, n, taus, dt):
        self.n = n
        self.rates = rates
        self.terms = taus
        self.dt = dt
        
        if len(self.terms) != self.rates.shape[1]:
            raise ValueError(
            f"Number of maturities (len(terms)={len(self.terms)}) "
            f"must equal number of observed rates (rates.shape[1]={self.rates.shape[1]})"
            )
    
    def _kappas(self, k1, b, mat=False):
        """Speed of factors.
        :param mat: if True computes kappas in the lower-triangular form, otherwise returns a vector
        """
        kappas = k1*b**(jnp.arange(start=1, stop=self.n+1)-1)
        if mat: return jnp.diag(kappas) + jnp.diag(-kappas[1:], k=-1)
        else: return kappas
    
    def _theta_q(self, theta_r, gamma, sigma, kappas):
        """risk-neutral drift adjustments"""
        return theta_r-gamma*sigma**2/kappas
    
    def _fx(self, x_prev, dt, params):
        """
        Latent state-space dynamics. Error matrix is dealt with inside the Unscent Kalman filter.
        :param params: State dynamic parameters in specific order, such that:
            element 0: k1,
            element 1: b,
            element 2:theta_r
        """
        k1, b, theta_r = params
        kappam = self._kappas(k1, b, mat=True)
        Phi = jax.scipy.linalg.expm(-dt*kappam)
        A = (jnp.identity(self.n) - Phi) @ jnp.full(self.n, theta_r)
        x_next = A + Phi @ x_prev
        
        return x_next
    
    def zcb_price(self, tau, k1, theta_r, sigma, kappas, gamma, x):
        
        n = self.n

        num = jnp.cumprod(kappas[::-1])[::-1]
        num_mat = num[:, None]
        idx = jnp.arange(n)
        J2 = idx[:, None, None]
        I2 = idx[None, :, None]
        M2 = idx[None, None, :]
        mask3 = (M2 >= J2) & (M2 != I2)
        D3 = kappas[M2] - kappas[I2]
        prod_diff = jnp.prod(jnp.where(mask3, D3, 1.0), axis=2)

        den = (kappas[None, :] * kappas[:, None]) * prod_diff

        alpha_mat = num_mat / den
        mask_load = idx[None, :] >= idx[:, None]
        alpha_mat = alpha_mat * mask_load

        M = 1.0 - jnp.exp(-kappas * tau)
        ploads = alpha_mat @ M

        G = tau - (1 - jnp.exp(-kappas * tau)) / kappas

        l1 = theta_r * k1 * jnp.dot(alpha_mat[:, 0], G)
        l2 = gamma * sigma**2 * jnp.sum(alpha_mat.T @ G)

        kappas_sum = kappas[:, None] + kappas[None, :]
        H = (
            tau
            - G[:, None]
            - G[None, :]
            + (1 - jnp.exp(-kappas_sum * tau)) / kappas_sum
        )                                                      

        quad = jax.vmap(lambda a_j: a_j @ (H @ a_j))(alpha_mat)
        l3 = 0.5 * sigma**2 * jnp.sum(quad)

        intercept = l1 - l2 - l3
        
        return jnp.exp(-jnp.dot(ploads, x) - intercept)

    def _hois(self, x, params):
        """Observation function using OIS rates.
        :param params: touple of parameters:
        """
        b, k1, theta_r, gamma, sigma = params
        kappas = self._kappas(k1, b)
        zcb_price_vmap = jax.vmap(lambda tau: self.zcb_price(tau, k1, theta_r, sigma, kappas, gamma, x))

        bond = zcb_price_vmap(self.terms)

        OIS = -(1/self.terms)*jnp.log(bond)
        return OIS

    def fit(self, init_param, maxiter=500, verb=True, method="Nelder-Mead"):

        def unpack_params(x):
            log_k1   = x[0]
            log_sigma = x[1]
            log_b_minus1 = x[2]
            theta_r = x[3]
            gamma = x[4]
            log_sigma_e = x[5]

            k1 = jnp.exp(log_k1)
            sigma = jnp.exp(log_sigma)
            sigma_e = jnp.exp(log_sigma_e)
            b = 1.0 + jnp.exp(log_b_minus1)

            return k1, sigma, b, theta_r, gamma, sigma_e

        def loss_fn(params):
            k1, sigma, b, theta_r, gamma, sigma_e = unpack_params(params)
            fx_params = (k1, b, theta_r)
            hx_params = (b, k1, theta_r, gamma, sigma)

            R = jnp.eye(self.rates.shape[1])*(sigma_e**2)
            Q = jnp.eye(self.n)

            alpha, beta, kappa = 1.0, 2.0, 0.0
            lam = alpha**2 * (self.n + kappa) - self.n
            c   = self.n + lam
            Wm0 = lam / c
            Wc0 = Wm0 + (1 - alpha**2 + beta)
            W_i = 1.0 / (2.0 * c)
            Wm = jnp.full((2*self.n + 1,), W_i).at[0].set(Wm0)
            Wc = jnp.full((2*self.n + 1,), W_i).at[0].set(Wc0)

            init_state = UKFState(x=jnp.zeros(self.n), P=jnp.eye(self.n))

            ukf_args = (self._fx, fx_params, self._hois, hx_params,
                Q, R, Wm, Wc, self.dt, c)
            
            def scan_fn(st, y):
                new_st, negll = ukf_step(st, y, *ukf_args)
                return new_st, (new_st, negll)
            
            _, (states, nlls) = jax.lax.scan(
                scan_fn,
                init_state,
                self.rates
            )
            
            return jnp.sum(nlls)

    def filter(self):
        """
        Runs the Unscented Kalman Filter over the observed rate series
        and returns the full sequence of filtered UKFState objects.
        """

        if not hasattr(self, 'fitted') or not self.fitted:
            raise RuntimeError("Model must be fitted before filtering.")

        Q = jnp.eye(self.n)
        R = jnp.eye(self.rates.shape[1]) * self.sigma_e**2

        alpha, beta, kappa = 1.0, 2.0, 0.0
        lam = alpha**2 * (self.n + kappa) - self.n
        c = self.n + lam
        Wm0 = lam / c
        Wc0 = Wm0 + (1 - alpha**2 + beta)
        Wi = 1 / (2 * c)
        Wm = jnp.full((2 * self.n + 1,), Wi).at[0].set(Wm0)
        Wc = jnp.full((2 * self.n + 1,), Wi).at[0].set(Wc0)
        init_state = UKFState(x=jnp.zeros(self.n), P=jnp.eye(self.n))
        fx_args = (self._fx, (self.k1, self.b, self.theta_r))
        hx_args = (self._hois, (self.b, self.k1, self.theta_r, self.gamma, self.sigma))

        def filter_step(state, y):
            new_state, _ = ukf_step(
                state, y,
                fx_args[0], fx_args[1],
                hx_args[0], hx_args[1],
                Q, R, Wm, Wc, self.dt, c
            )
            return new_state, new_state

        _, filtered_states = jax.lax.scan(
            filter_step,
            init_state,
            self.rates
        )

        return filtered_states

    def forecast_zcb(self, x0: jnp.ndarray, steps: int) -> jnp.ndarray:
        """
        Forecast zero-coupon bond prices for `self.terms` maturities, over `steps`
        future time‐points, starting from latent state `x0`.
        Returns an array of shape (steps, n_maturities).
        """
        
        fx_params = (self.k1, self.b, self.theta_r)

        def one_step(x_prev, _):
            x_next = self._fx(x_prev, self.dt, fx_params)

            kappas = self._kappas(self.k1, self.b)
            zcb_one = jax.vmap(
                lambda tau: self.zcb_price(
                    tau,
                    self.k1,
                    self.theta_r,
                    self.sigma,
                    kappas,
                    self.gamma,
                    x_next
                )
            )(self.terms)

            return x_next, zcb_one

        _, zcb_paths = jax.lax.scan(
            one_step,
            x0,
            None,
            length=steps
        )
        return zcb_paths