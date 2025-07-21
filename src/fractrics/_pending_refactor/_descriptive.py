from fractrics._deprecated._pending_refactor._helper import OLS, partition_map
import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.stats import norm, chi2, t, f

def z_score(array):
    """ 
    Computes the Z-score for an input array
    """
    
    # ensure to work with a numpy array
    array = np.array(array)
    return (array - np.nanmean(array))/np.nanstd(array)

def kurtosis(array):
    """
    Computes kurtosis of input array
    """
    return (z_score(array)**4).mean()

def skewness(array):
    """
    Computes skewness of input array
    """

    return (z_score(array)**3).mean()

def acf(ts):
    """
    Compute autocorrelation, T-ratio and Ljung-Box and their p-values for all lags of a time series.
    Parameters:
        ts: Time series data.

    Returns:
        acf (ndarray): Autocorrelation values and statistics for lags 0 to n-1.
    """
    
    ts = np.asarray(ts)
    n = len(ts)
    lags = range(n)
    acf = (np.correlate(ts - np.nanmean(ts), ts - np.nanmean(ts), mode='full'))[n-1:] / (np.nanvar(ts)*n)
    t_ratio = acf/(np.sqrt(1+2*np.cumsum(acf**2)))
    pv_t = 2 * (1 -norm.cdf(abs(t_ratio)))
    q_st = n*(n+2)*np.cumsum(acf**2/np.arange(n, 0, -1))
    pv_q = 1 - chi2.cdf(q_st, lags)
        
    acf_df = pd.DataFrame({
    'lag': lags,
    'acf': acf,
    't_ratio': t_ratio,
    'pv_t' : pv_t,
    'q_stat': q_st,
    'pv_q': pv_q
    })
    
    return acf_df

def rs(ts, logc=0, even=True):
    """
    Computes the rescaled range statistic for a time series.
    
    :param ts: the series to be analized
    :param even: whether to compute the rescaled range only on evenly split partitions
    :param logc: specifies the order of the log change of the series to be used. (0: no log change, 1: first order log change)
    """
    ts = np.array(ts) # making sure the series is a numpy array
    
    ts_adj = np.array(np.log(ts[1:] / (ts[:-1]**logc))) #transform in logaritmic scale
    
    # all lengths of partitions up to 2 that divide the series in equal parts
    if even: n = np.array([n for n in range(2, ((len(ts)-1)//2)+1) if (len(ts)-1) % n == 0])
    else: n = np.arange(2, ((len(ts)-1)//2)+1)
    
    rs = np.zeros(len(n)) #initialize vector of R/S
    r = np.zeros(len(n)) #initialize vector of adjusted range
    
    # expected statistics of a brownian motion series according to Anis and Lloyd
    i = np.arange(1, n[-1])
    e_rs = (1/np.pi)*(np.exp(sp.gammaln(0.5*(n-1))-sp.gammaln(0.5*n)))*np.array([np.sum(((k - i[:k-1]) / i[:k-1]) ** 0.5) for k in n])

    def rs_stat(array):
        
        # cumulative deviance from mean
        cum_dev = np.cumsum(array - np.nanmean(array))

        return (np.nanmax(cum_dev)-np.nanmin(cum_dev))/ np.nanstd(array)
    
    def r_stat(array):
    
        # cumulative deviance from mean
        cum_dev = np.cumsum(array - np.nanmean(array))

        return (np.nanmax(cum_dev)-np.nanmin(cum_dev))
    
    for j,i in enumerate(n):
        # each element is the average R/S statistic for the corresponding number of days
        rs[j] = np.nanmean(partition_map(arr=ts_adj, func=rs_stat, num_partitions=i))
        r[j] = np.nanmean(partition_map(arr=ts_adj, func=r_stat, num_partitions=i))
        
    # note: (number of partitions) = (size of the ts) / (days in each partitions)
    # since n contains all possible integer numbers of partitions, it is the inverse of an array containing the number of days in each partitions
    # therefore n is inverted to be used in a log plot. The same is dove for e_rs
    
    rs_df = pd.DataFrame({
        'n_days': n[::-1], # reverse the array to resemble the number of days
        'log_days': np.log(n[::-1]),
        
        'rs': rs,
        'e_rs': e_rs[::-1],
        'loge_rs': np.log(e_rs[::-1]),
        'log_rs': np.log(rs),
        'v_stat': rs / np.sqrt(n[::-1]),
        
        'r':r,
        'e_r': np.sqrt(n[::-1]*np.pi/2),
        'var_r': np.sqrt(n[::-1]*(np.pi**2/6 - np.pi/2)),
        'z_r': (r - np.sqrt(n[::-1]*np.pi/2))/np.sqrt(n[::-1]*(np.pi**2/6 - np.pi/2)),
        'pv_r': 2*(1- norm.cdf(abs((r - np.sqrt(n[::-1]*np.pi/2))/np.sqrt(n[::-1]*(np.pi**2/6 - np.pi/2)))))
        })
    
    return rs_df

def hurst(ts, logc=0, even=True):
    """
    Returns the hurst exponent of a series, its gaussian z-score and p-value,
    and the minimum number of observations to consider the result significant at the 5% 
    confidence interval
    """
    ts_rs = rs(ts=ts, logc=logc, even=even)
    h = OLS(x=ts_rs['log_days'], y=ts_rs['log_rs'], intercept=True)
    e_h = OLS(x=ts_rs['log_days'], y=ts_rs['loge_rs'], intercept=True)
   
    # minimum observations to consider an H != 0.5 significant
    min_T = 4/((h[0]-e_h[0])**2)
    gauss_z = (h[0]-e_h[0])*np.sqrt(len(ts))
    pv_h = 2 *(1 - norm.cdf(gauss_z))
    
    #OLS predictions and residuals
    h_fit = ts_rs['log_days']*h[0]+h[1]
    h_resid = ts_rs['log_rs'] - h_fit
    
    #variance of the residuals of an OLS: -2 parameters H and c
    deg_freedom = (len(h_resid)-2)
    resid_var = np.sum(h_resid**2)/deg_freedom
    
    #standard error of each parameter
    std_err = np.sqrt(resid_var/h)
    
    # T statistics and p. value
    t_stat = h/std_err
    pv_t = 2 * (1 - t.cdf(np.abs(t_stat), deg_freedom))
    
    # F statistics   
    f_stat = np.sum((h_fit - ts_rs['log_rs'].mean())**2)/resid_var
    pv_f = f.cdf(f_stat, 1, deg_freedom)
    
    hurst_={
        "hurst": h[0],
        "c": h[1],
        "pv_hurst": pv_h,
        "min_obs": min_T,
        "pv_t":pv_t,
        "pv_F":pv_f
    }
    
    return hurst_
