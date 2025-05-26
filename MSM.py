import jax
import itertools
import numpy as np
import jax.numpy as jnp
from jaxopt import LBFGSB
from functools import partial
import jax.scipy.stats as jss
from jaxopt import ScipyMinimize
from scipy.stats import rv_discrete
from jax.scipy.special import logsumexp
from fractrics.diagnostics import pareto_shape, tail_pv

def sim_MSM(support, prob, sigma_l, b, gamma_k, k, T):
    """
    Simulates a Markov Switching Multifractal 
    """
    support = np.array(support)
    prob = np.array(prob)
    
    # # check that support and probability are of the same length
    # if len(support) != len(prob): raise ValueError("Support and prob vectors must be of the same length")
    
    # # check that if distribution M has positive support and unit mean
    # if support.any()<0: raise ValueError("Support elements must be positive")
    # if np.sum(support * prob) != 1: raise ValueError("M must have unit mean")
    
    # construct the elements for the simulation
    M = rv_discrete(values=(support, prob))
    gamma = 1 - (1 - gamma_k) ** (1 / (b ** (np.arange(k, 0, -1) - 1)))
    
    #initialize simulated state and series vectors
    r_sim = np.zeros(T)
    v_sim = np.zeros(T)
    state = M.rvs(size=k)
    
    #simulation
    for t in range(T):
        
        # draw which of the states will transition at time t
        switch = np.random.rand(len(gamma)) < gamma
        
        # using mask switch, draw from M only the new states
        state[switch] = M.rvs(size=np.sum(switch))
        
        # the simulated volatility and return are
        v_sim[t]=sigma_l*np.sqrt(np.prod(state))
        r_sim[t]=v_sim[t]*np.random.standard_normal()
        
    return r_sim, v_sim

class MSM:
    
    def __init__(self, ts, k=None, pM=None):
        """
        Markov Switching  Multifractal initialization.

        :param ts: time series data (array-like)
        :param k: number of possible states
        :param pM: probability vector of the discrete marginal distribution of multipliers. Default is symmetric Bernoulli.
        
        """
        self.ts = jnp.array(ts)
        self.k = 2 if k is None else k
        self.pM = jnp.repeat(0.5, 2) if pM is None else jnp.array(pM)
        self.fitted = False

    @partial(jax.jit, static_argnames=["self"])
    def ergotic_distr(self, support):
        """
        Log of Ergodic distribution and possible states (non log).
        States is the k-th repeated cartesian product of support
        The ergotic distribution is vector resulting from row-multiplication of the k-th cartesian product of pM
        """
        states = jnp.array(list(itertools.product(support, repeat=self.k)))
        prob_cp = jnp.array(list(itertools.product(self.pM, repeat=self.k)))
        probs = jnp.prod(prob_cp, axis=1).reshape([len(self.pM)] * self.k)
        return states, jnp.log(probs)
    
    @partial(jax.jit, static_argnames=["self"])
    def w(self, x, states, sigma_l):
        """Likelihood vector for observed data and state permutations."""
        sigma = sigma_l * jnp.sqrt(jnp.prod(states, axis=1))
        return jss.norm.logpdf(x.reshape(-1, 1), loc=0, scale=sigma)

    @partial(jax.jit, static_argnames=["self"])
    def gamma_prob(self, b, gamma_k):
        """Transition probabilities for each k."""
        return 1 - (1 - gamma_k) ** (1 / (b ** (jnp.arange(self.k, 0, -1) - 1)))

    @partial(jax.jit, static_argnames=["self"])
    def A_tensor(self, gamma_vec):
        """
        Computes the transition tensor of the chain. 
        Each element is P[M_(k, t) = j | M_(k, t-1) = i]
        """
        A_ten = jnp.log(jnp.array([(1-g)*jnp.eye(len(self.pM)) + g*jnp.tile(self.pM, (len(self.pM), 1)) for g in gamma_vec]))
        return A_ten

    @partial(jax.jit, static_argnames=["self", "comp"])
    def update_distr(self, sigma_l, b, gamma_k, support, newd=None, comp=True, pi_start=None):
        """
        Recursively compute the state probabilities over time and the NLL of the chain.

        :param gamma_k: transition probability base
        :param b: scaling parameter for transition probabilities
        :param sigma_l: volatility scalar
        :param newd: (used only if fit=True) new data to update the distribution to (for online learning)
        :param comp: specifies whether to return only the likelihood (comp=False)
        :param pi_start: tensor from which to start the update
        :return: final state probability tensor, previous state tensors, Log Likelihood (LL)
        """        
        # out of sample forecasting case
        if self.fitted:
            # for the first iteration
            if pi_start is None: pi_t = self.log_pi_t
            else: pi_t = pi_start
            
            states = self.states
            w_val = self.w(jnp.array(newd), states, sigma_l)
            gamma_vec = self.gamma_vec
            A_tens = self.A_tens
        
        #otherwise the model is initialized
        else:
            # create the ergotic distribution to a tensor
            states, pi_t = self.ergotic_distr(support)
            #compute observation likelihood matrix
            w_val = self.w(self.ts[1:], states, sigma_l)
            # Compute the transition tensor
            gamma_vec = self.gamma_prob(b, gamma_k)
            A_tens = self.A_tensor(gamma_vec)

        # Initialize the carry (state probabilities at t=0 and initial NLL)
        carry_init = (pi_t, 0.0)        
        
        def transition_tensor(carry, w_t):
            pi_t, LL = carry

            # Apply the transition tensor to get the updated state probabilities
            for k in range(A_tens.shape[0]):
                # Move mode-k axis to front
                log_A_k = A_tens[k].T
                pi_t = jnp.moveaxis(pi_t, k, 0)

                # For each j, sum over i: logsumexp(log_A_k[i, j] + log_pi_t[i])
                pi_t = jax.vmap(  
                    lambda log_col: logsumexp(log_col[:, None] + pi_t, axis=0)
                )(log_A_k)

                # Move the axis back
                pi_t = jnp.moveaxis(pi_t, 0, k)
                
            pi_adj = pi_t.flatten() + w_t
            LL_t = -logsumexp(pi_adj)
            
            # Normalize the updated state probabilities
            pi_t_next = (pi_adj + LL_t).reshape([len(self.pM)] * self.k)
            
            # Carry forward updated state probabilities
            return (pi_t_next, LL + LL_t), pi_t_next
            
        # Perform the scan operation over the likelihoods
        (pi_t, fin_LL), pi_list = jax.lax.scan(transition_tensor, carry_init, w_val)
        
        if comp: return pi_t, pi_list, fin_LL
        else: return fin_LL
    
    def fit(self, init_param, maxiter=500, verb=True, method="Nelder-Mead"):
        """
        init_param: 
            log_sigma_l,               # scalar
            log_b_minus1,              # scalar
            logit_gamma_K,             # scalar (for the largest state K)
            *log_m_raw                 # length = len(self.pM)
        """

        M = len(self.pM)

        def unpack_params(x):
            #slice out params
            log_sigma_l   = x[0]
            log_b_minus1  = x[1]
            logit_gamma_K = x[2]
            log_m_raw     = x[3:]

            #back‐transform
            sigma_l = jnp.exp(log_sigma_l)
            b       = 1.0 + jnp.exp(log_b_minus1)
            gamma_k = jax.nn.sigmoid(logit_gamma_K)

            # discrete‐support D: softmax then rescale to unit mean
            m_pos = jnp.exp(log_m_raw)
            support = m_pos / jnp.dot(self.pM, m_pos)

            return gamma_k, b, sigma_l, support

        def loss_fn(x):
            gamma_k, b, sigma_l, support = unpack_params(x)
            ll = self.update_distr(sigma_l, b, gamma_k, support, comp=False)
            return ll

        if method == "LBFGSB":
            bounds = (jnp.repeat(-100, M+3), jnp.repeat(100, M+3))
            solver = LBFGSB(fun=loss_fn, maxiter=maxiter, verbose=verb)
            result = solver.run(init_param, bounds=bounds)
            self.converged = result.state.error < solver.tol
            if not self.converged: print("Warning! Optimization did not converge.")
            
        elif method == "Nelder-Mead":
            solver = ScipyMinimize(method="Nelder-Mead",fun=loss_fn, maxiter=maxiter, options={"disp": verb})
            result = solver.run(init_param)
            self.converged = result.state.success
            if not self.converged: print("Warning! Optimization did not converge.")
            self.n_iter = result.state.iter_num
            
        elif method == "differential_evolution":
            from scipy.optimize import differential_evolution

            # Define bounds for each parameter (adjust as needed)
            bounds = [(-5, 5),    # log_sigma_l
                      (-5, 5),    # log_b_minus1
                      (-5, 5)]    # logit_gamma_K
            bounds += [(-5, 5)] * M  # log_m_raw for each support

            # DE expects numpy arrays, so wrap loss_fn
            def loss_fn_np(x):
                x = jnp.array(x)
                return float(loss_fn(x))

            result = differential_evolution(loss_fn_np, bounds, maxiter=maxiter, disp=verb)
            self.converged = result.success
            if not self.converged:
                print("Warning! Optimization did not converge.")
            # Mimic the result object for consistency
            class DummyResult:
                pass
            dummy = DummyResult()
            dummy.params = jnp.array(result.x)
            self.result = dummy
            result = dummy
        
        self.result = result
        
        self.gamma_k, self.b, self.sigma_l, self.support = unpack_params(result.params)
        self.log_pi_t, self.pi_list, self.LL = self.update_distr(self.sigma_l, self.b, self.gamma_k, self.support)
        
        # exit log world and flatten probabilitis
        self.pi_list = jnp.exp(jnp.stack([pi.flatten() for pi in self.pi_list]))
        
        self.pi_t = jnp.exp(self.log_pi_t)
        self.gamma_vec = self.gamma_prob(self.b, self.gamma_k)
        self.A_tens = self.A_tensor(self.gamma_vec)
        
        self.states = jnp.array(list(itertools.product(self.support, repeat=self.k)))
        #Maximized posterior marginals corresponding states (different from Viterbi-implied states)
        self.MPM_states = self.states[jnp.argmax(self.pi_list, axis=1)]
        
        self.sigma = self.sigma_l * jnp.sqrt(jnp.prod(self.states, axis=1))
        self.fitted = True
    
    def forecast(self, step=1, insample=True, new_lpi=None):
        """
        Returns expected volatiliy (standard deviation) after step periods of time.
        It is then possible to simulate returns by sampling from N(0, exp_std)
            :param step: How many time units into the future to forecast
            :param insample: Specifies whether fitted distribution needs to be used.
            :param new_lpi: if insample is false, specifies the new pi_t to use
        """
        
        if self.fit:
            
            # initialize the predictive distribution as current distribution
            if insample: log_pi_forecast = self.log_pi_t
            else: log_pi_forecast = new_lpi
                
            for _ in range(step):
                
                # Apply the transition tensor to get the forecast probabilities
                for k in range(self.A_tens.shape[0]):
                    # Move mode-k axis to front
                    log_A_k = self.A_tens[k].T
                    log_pi_forecast = jnp.moveaxis(log_pi_forecast, k, 0) 

                    # For each j, sum over i: logsumexp(log_A_k[i, j] + log_pi_t[i])
                    log_pi_forecast = jax.vmap(  # over j in log_A_k.T
                        lambda log_col: logsumexp(log_col[:, None] + log_pi_forecast, axis=0)
                    )(log_A_k)
                    
                    log_pi_forecast = jnp.moveaxis(log_pi_forecast, 0, k)
            
            # flatten and exp() to obtain the probability vector
            pi_forecast = jnp.exp(log_pi_forecast.flatten())
            return jnp.sum(pi_forecast * self.sigma), pi_forecast
        
        else: raise ValueError("Fit model first.")
        
    def bayes_filter(self, newd, step=None):
        """
        Makes 1-step ahead out of sample forecast and then updates distribution with new data.
            :param newd: Either an array-like or "self". If "self", it computes a path-dependent simulation.
        """
        new_lpi= self.log_pi_t
        impl_noise = []
        vol = []
        
        for y in newd:
            sigma_t, _ = self.forecast(insample=False, new_lpi=new_lpi)
            vol.append(sigma_t)
            #model-implied noise
            impl_noise.append(y/sigma_t)
            new_lpi, _, _ = self.update_distr(self.sigma_l, self.b, self.gamma_k, self.support, newd=y, pi_start = new_lpi)
            
        return vol, impl_noise
    
    def mc_forecast(self, steps, n_sim, key):
        """
        Fully JAX-compatible Monte Carlo forecast.
        Simulates n_sim independent paths, each of length 'steps'.
        Returns arrays of shape (n_sim, steps) for simulated returns and volatilities.
        """
        def single_path(carry, key):
            new_lpi = carry
            # Forecast volatility for this step
            sigma_t, _ = self.forecast(insample=False, new_lpi=new_lpi)
            # Draw standard normal noise
            noise = jax.random.normal(key)
            x_t = sigma_t * noise
            # Update distribution with new simulated data
            new_lpi, _, _ = self.update_distr(self.sigma_l, self.b, self.gamma_k, self.support, newd=x_t, pi_start=new_lpi)
            return new_lpi, (sigma_t, x_t)

        def simulate_path(key):
            keys = jax.random.split(key, steps)
            # Run scan for one path
            _, (vols, xs) = jax.lax.scan(single_path, self.log_pi_t, keys)
            return vols, xs

        # Split the main key for each simulation
        sim_keys = jax.random.split(key, n_sim)
        # Vectorize over simulations
        vols, xs = jax.vmap(simulate_path)(sim_keys)
        return vols, xs
            
    def tail_idx(self, nsim = 10):
        """
        Computes the tail index of the time series and of model simulations
        """
        if self.fitted:
            self.tail = pareto_shape(self.ts)
            
            #initialize list of simulation tails
            sim_tail = jnp.zeros(shape=[nsim, len(self.tail)])
            
            # Compute the pareto tails for each simulation and store in each row of sim_tail
            for i in range(nsim):
                r_sim, _ = sim_MSM(self.support, self.pM, self.sigma_l, self.b, self.gamma_k, self.k, len(self.ts))
                sim_tail = sim_tail.at[i].set(pareto_shape(r_sim))
                
            # compute the average of each tail order
            sim_tail = jnp.mean(sim_tail, axis=0)
            self.tail_pv = tail_pv(self.tail, sim_tail)
            
        else: raise ValueError("Fit model first.")
        
def sim_levelMSM(support, prob, sigma_l, b, gamma_k, k, beta0, beta1, eta, T, init):
    """
    Simulates a Markov Switching Multifractal with level effect
    """
    support = np.array(support)
    prob = np.array(prob)
    
    # construct the elements for the simulation
    M = rv_discrete(values=(support, prob))
    gamma = 1 - (1 - gamma_k) ** (1 / (b ** (np.arange(k, 0, -1) - 1)))
    
    #initialize simulated state and series vectors
    r_sim = np.zeros(T+1)
    x_sim = np.zeros(T)
    v_sim = np.zeros(T)
    state = M.rvs(size=k)
    r_new = init
    #simulation
    for t in range(T):
        r_sim[t] = r_new
        # draw which of the states will transition at time t
        switch = np.random.rand(len(gamma)) < gamma
        
        # using mask switch, draw from M only the new states
        state[switch] = M.rvs(size=np.sum(switch))
        
        # the simulated volatility and return are
        v_sim[t]=sigma_l*np.sqrt(np.prod(state))
        x_sim[t]=v_sim[t]*np.random.standard_normal()
        
        r_new = beta0 + r_sim[t]*(x_sim[t]*r_sim[t]**(eta-1) + 1 + beta1)
        
    return r_sim, v_sim

class level_MSM:
    
    def __init__(self, ts, k=None, pM=None):
        """
        Level-Markov Switching  Multifractal initialization.

        :param ts: time series data (array-like)
        :param k: number of possible states
        :param pM: probability vector of the discrete marginal distribution of multipliers. Default is symmetric Bernoulli.
        
        """
        self.ts = jnp.array(ts)
        self.dts = self.ts[1:]-self.ts[:-1]
        self.k = 2 if k is None else k
        self.pM = jnp.repeat(0.5, 2) if pM is None else jnp.array(pM)
        self.fitted = False

    @partial(jax.jit, static_argnames=["self"])
    def ergotic_distr(self, support):
        """
        Log of Ergodic distribution and possible states (non log).
        States is the k-th repeated cartesian product of support
        The ergotic distribution is vector resulting from row-multiplication of the k-th cartesian product of pM
        """
        states = jnp.array(list(itertools.product(support, repeat=self.k)))
        prob_cp = jnp.array(list(itertools.product(self.pM, repeat=self.k)))
        probs = jnp.prod(prob_cp, axis=1).reshape([len(self.pM)] * self.k)
        return states, jnp.log(probs)
    
    @partial(jax.jit, static_argnames=["self"])
    def w(self, x, states, sigma_l):
        """Likelihood vector for observed data and state permutations."""
        sigma = sigma_l * jnp.sqrt(jnp.prod(states, axis=1))
        return jss.norm.logpdf(x.reshape(-1, 1), loc=0, scale=sigma)

    @partial(jax.jit, static_argnames=["self"])
    def gamma_prob(self, b, gamma_k):
        """Transition probabilities for each k."""
        return 1 - (1 - gamma_k) ** (1 / (b ** (jnp.arange(self.k, 0, -1) - 1)))

    @partial(jax.jit, static_argnames=["self"])
    def A_tensor(self, gamma_vec):
        """
        Computes the transition tensor of the chain. 
        Each element is P[M_(k, t) = j | M_(k, t-1) = i]
        """
        A_ten = jnp.log(jnp.array([(1-g)*jnp.eye(len(self.pM)) + g*jnp.tile(self.pM, (len(self.pM), 1)) for g in gamma_vec]))
        return A_ten

    @partial(jax.jit, static_argnames=["self"])
    def _x(self, beta0, beta1, eta, newd=None, oldd=None):
        """
        Model-implied multifractal component in log-space.
        """
        if newd is None:
            # Sign decomposition trick to keep the log consistent
            den = self.ts[1:]
            num = self.dts - beta0 - beta1 * self.ts[1:]
            sign = jnp.sign(num)*jnp.sign(den)
            
        else:
            # ensure new and old observation are arrays even if only 1 value is passed.
            newd = jnp.atleast_1d(newd)
            oldd = jnp.atleast_1d(oldd)
            
            # if the above have 1 element = oldd, else concatenates last oldd to all but last newd
            den = jnp.concatenate([oldd[-1][None], newd[:-1]])
            new_dts = newd - den          
            num = new_dts - beta0 - beta1 * den
            sign = jnp.sign(num)*jnp.sign(den)
        
        x = sign * jnp.exp(jnp.log(jnp.abs(num)) - eta*jnp.log(jnp.abs(den)))
        return x
    
    @partial(jax.jit, static_argnames=["self", "comp"])
    def update_distr(self, sigma_l, b, gamma_k, support, beta0, beta1, eta, newd=None, oldd=None, comp=True, pi_start=None):
        """
        Recursively compute the state probabilities over time and the NLL of the chain.

        :param gamma_k: transition probability base
        :param b: scaling parameter for transition probabilities
        :param sigma_l: volatility scalar
        :param newd: (used only if fit=True) new data to update the distribution to (for online learning)
        :param comp: specifies whether to return only the likelihood (comp=False)
        :param pi_start: tensor from which to start the update
        :return: final state probability tensor, previous state tensors, Log Likelihood (LL)
        """
         
        def transition_tensor(carry, wr):
            w_t, re_t = wr
            pi_t, LL = carry

            # Apply the transition tensor to get the updated state probabilities
            for k in range(A_tens.shape[0]):
                # Move mode-k axis to front
                log_A_k = A_tens[k].T
                pi_t = jnp.moveaxis(pi_t, k, 0)

                # For each j, sum over i: logsumexp(log_A_k[i, j] + log_pi_t[i])
                pi_t = jax.vmap(  
                    lambda log_col: logsumexp(log_col[:, None] + pi_t, axis=0)
                )(log_A_k)

                # Move the axis back
                pi_t = jnp.moveaxis(pi_t, 0, k)
                
            pi_adj = pi_t.flatten() + w_t
            
            #differently from MSM, the likelihood does not scale the distribution directly
            den = -logsumexp(pi_adj)
            LL_t = den - re_t
            
            # Normalize the updated state probabilities
            pi_t_next = (pi_adj + den).reshape([len(self.pM)] * self.k)
            
            # Carry forward updated state probabilities
            return (pi_t_next, LL + LL_t), pi_t_next
        
        # out of sample forecasting case
        if self.fitted:
            # for the first iteration
            if pi_start is None: pi_t = self.log_pi_t
            else: pi_t = pi_start
            r_eta = jnp.log(eta)
            x = self._x(self.beta0, self.beta1, self.eta, newd, oldd)
            states = self.states
            w_val = self.w(x, states, sigma_l)
            gamma_vec = self.gamma_vec
            A_tens = self.A_tens
            
        #otherwise the model is initialized
        else:
            # create the ergotic distribution to a tensor
            states, pi_t = self.ergotic_distr(support)
            # compute multifractal component
            x = self._x(beta0, beta1, eta)
            # compute level component in log-space
            r_eta = jnp.log(self.ts[1:]**eta)
            #compute observation likelihood matrix
            w_val = self.w(x, states, sigma_l)
            # Compute the transition tensor
            gamma_vec = self.gamma_prob(b, gamma_k)
            A_tens = self.A_tensor(gamma_vec)

        if (jnp.shape(w_val) == () or jnp.shape(w_val)[0] == 1):
            # Single observation: call transition_tensor once
            (pi_t, fin_LL), pi_list = transition_tensor((pi_t, 0.0), (w_val, r_eta))
            pi_list = jnp.expand_dims(pi_list, 0)  # to keep output shape consistent
        else:
            # Sequence: use scan
            (pi_t, fin_LL), pi_list = jax.lax.scan(transition_tensor, (pi_t, 0.0), (w_val, r_eta))
                     
        if comp: return pi_t, pi_list, fin_LL
        else: return fin_LL
    
    def fit(self, init_param, maxiter=500, verb=True, method="Nelder-Mead"):
        """
        init_param: 
            log_sigma_l,               # scalar
            log_b_minus1,              # scalar
            logit_gamma_K,             # scalar (for the largest state K)
            log_eta,                   # scalar
            beta0,                     # scalar, unconstrained
            beta1,                     # scalar, unconstrained
            *log_m_raw                 # length = len(self.pM)
        """

        M = len(self.pM)

        def unpack_params(x):
            #slice out params
            log_sigma_l   = x[0]
            log_b_minus1  = x[1]
            logit_gamma_K = x[2]
            log_eta       = x[3]
            beta0         = x[4]
            beta1         = x[5]
            log_m_raw     = x[6:]

            #back‐transform
            sigma_l = jnp.exp(log_sigma_l)
            b       = 1.0 + jnp.exp(log_b_minus1)
            gamma_k = jax.nn.sigmoid(logit_gamma_K)
            eta = jnp.exp(log_eta)
            m_pos = jnp.exp(log_m_raw)
            support = m_pos / jnp.dot(self.pM, m_pos)

            return gamma_k, b, sigma_l, beta0, beta1, eta, support

        def loss_fn(x):
            gamma_k, b, sigma_l, beta0, beta1, eta, support = unpack_params(x)
            # Defensive checks for invalid values
            if (
                not jnp.all(jnp.isfinite(support)) or
                not jnp.all(support > 0) or
                not jnp.isfinite(sigma_l) or sigma_l <= 0 or
                not jnp.isfinite(b) or b <= 1 or
                not jnp.isfinite(eta) or eta <= 0 or
                not jnp.isfinite(beta0) or
                not jnp.isfinite(beta1)
            ):
                return 1e10
            ll = self.update_distr(sigma_l, b, gamma_k, support, beta0, beta1, eta, comp=False)
            if not jnp.isfinite(ll):
                return 1e10
            return ll

        if method == "LBFGSB":
            bounds = (jnp.repeat(-100, M+3), jnp.repeat(100, M+3))
            solver = LBFGSB(fun=loss_fn, maxiter=maxiter, verbose=verb)
            result = solver.run(init_param, bounds=bounds)
            self.converged = result.state.error < solver.tol
            if not self.converged: print("Warning! Optimization did not converge.")
            
        elif method == "Nelder-Mead":
            solver = ScipyMinimize(method="Nelder-Mead",fun=loss_fn, maxiter=maxiter, options={"disp": verb})
            result = solver.run(init_param)
            self.converged = result.state.success
            if not self.converged: print("Warning! Optimization did not converge.")
            self.n_iter = result.state.iter_num
        
        elif method == "differential_evolution":
            from scipy.optimize import differential_evolution

            # Define bounds for each parameter (adjust as needed)
            bounds = [
                (-10, 5),   # log_sigma_l
                (-10, 5),   # log_b_minus1
                (-5, 5),    # logit_gamma_K
                (-5, 5),    # log_eta
                (-10, 10),  # beta0
                (-10, 10)   # beta1
            ]
            bounds += [(-5, 5)] * M  # log_m_raw for each support

            # DE expects numpy arrays, so wrap loss_fn
            def loss_fn_np(x):
                x = jnp.array(x)
                return float(loss_fn(x))

            result = differential_evolution(loss_fn_np, bounds, maxiter=maxiter, disp=verb)
            self.converged = result.success
            if not self.converged:
                print("Warning! Optimization did not converge.")
            # Mimic the result object for consistency
            class DummyResult:
                pass
            dummy = DummyResult()
            dummy.params = jnp.array(result.x)
            self.result = dummy
            result = dummy
        
        self.result = result
        
        self.gamma_k, self.b, self.sigma_l, self.beta0, self.beta1, self.eta, self.support = unpack_params(result.params)
        self.log_pi_t, self.pi_list, self.LL = self.update_distr(self.sigma_l, self.b, self.gamma_k, self.support, self.beta0, self.beta1, self.eta)
        
        # exit log world and flatten probabilitis
        self.pi_list = jnp.exp(jnp.stack([pi.flatten() for pi in self.pi_list]))
        
        self.pi_t = jnp.exp(self.log_pi_t)
        self.gamma_vec = self.gamma_prob(self.b, self.gamma_k)
        self.A_tens = self.A_tensor(self.gamma_vec)
        self.x = self._x(self.beta0, self.beta1, self.eta)
        self.states = jnp.array(list(itertools.product(self.support, repeat=self.k)))
        #Maximized posterior marginals corresponding states (different from Viterbi-implied states)
        self.MPM_states = self.states[jnp.argmax(self.pi_list, axis=1)]
        
        self.sigma = self.sigma_l * jnp.sqrt(jnp.prod(self.states, axis=1))
        self.fitted = True
    
    def forecast(self, step=1, insample=True, new_lpi=None, newd=None):
        """
        Returns expected volatiliy (standard deviation) after step periods of time.
        It is then possible to simulate returns by sampling from N(0, exp_std)
            :param step: How many time units into the future to forecast
            :param insample: Specifies whether fitted distribution needs to be used.
            :param new_lpi: if insample is false, specifies the new pi_t to use
        """
        
        if self.fit:
            
            if insample: 
            # initialize the predictive distribution as current distribution
                log_pi_forecast = self.log_pi_t
            #first observation is last of original data
                r_n = self.ts[-1]
            else: 
                log_pi_forecast = new_lpi
                r_n = newd
            
            #initialize vector of new data
            r_new = jnp.zeros(step)    
            
            for h in range(step):
                # Apply the transition tensor to get the forecast probabilities
                for k in range(self.A_tens.shape[0]):
                    # Move mode-k axis to front
                    log_A_k = self.A_tens[k].T
                    log_pi_forecast = jnp.moveaxis(log_pi_forecast, k, 0) 

                    # For each j, sum over i: logsumexp(log_A_k[i, j] + log_pi_t[i])
                    log_pi_forecast = jax.vmap(  # over j in log_A_k.T
                        lambda log_col: logsumexp(log_col[:, None] + log_pi_forecast, axis=0)
                    )(log_A_k)
                    
                    log_pi_forecast = jnp.moveaxis(log_pi_forecast, 0, k)
            
                # flatten and exp() to obtain the probability vector
                pi_forecast = jnp.exp(log_pi_forecast.flatten())
                x_new = jnp.sum(pi_forecast * self.sigma)
                r_n = self.beta0 + self.beta1 * r_n+ x_new*r_n**self.eta + r_n
                
                #apply change
                r_new.at[h].set(r_n)
            
            return r_n, x_new
        
        else: raise ValueError("Fit model first.")
        
    def bayes_filter(self, newd):
        """
        Makes 1-step ahead out of sample forecast and then updates distribution with new data.
        """
        new_lpi = self.log_pi_t
        x_list = []
        r_list = []
        oldd = self.ts[-1]
        
        for y in newd:
            # Update state with new observation first
            new_lpi, _, _ = self.update_distr(self.sigma_l, self.b, self.gamma_k, self.support, self.beta0, self.beta1, self.eta, newd=y, oldd=oldd, pi_start=new_lpi)
            # Forecast using the updated state
            r_t, x_new = self.forecast(insample=False, new_lpi=new_lpi, newd=y)
            r_list.append(r_t)
            x_list.append(x_new)
            oldd = r_t
        return r_list, x_list
            
    def tail_idx(self, nsim = 10):
        """
        Computes the tail index of the time series and of model simulations
        """
        if self.fitted:
            self.tail = pareto_shape(self.ts)
            
            #initialize list of simulation tails
            sim_tail = jnp.zeros(shape=[nsim, len(self.tail)])
            
            # Compute the pareto tails for each simulation and store in each row of sim_tail
            for i in range(nsim):
                r_sim, _ = sim_levelMSM(self.support, self.pM, self.sigma_l, self.b, self.gamma_k, self.k, self.beta0, self.beta1, self.eta, len(self.ts))
                sim_tail = sim_tail.at[i].set(pareto_shape(r_sim))
                
            # compute the average of each tail order
            sim_tail = jnp.mean(sim_tail, axis=0)
            self.tail_pv = tail_pv(self.tail, sim_tail)
            
        else: raise ValueError("Fit model first.")