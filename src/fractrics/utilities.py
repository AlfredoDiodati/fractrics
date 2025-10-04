import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from fractrics._components.core import ts_metadata

def summary(model:ts_metadata):
    table = {'parameters': model.parameters}
    if hasattr(model, "standard_errors"): table['standard_errors'] = model.standard_errors
    if hasattr(model, "robust_standard_errors"): table['robust_standard_errors'] = model.robust_standard_errors
    table = pd.DataFrame(table)
    print(table)
    print(pd.Series(model.optimization_info))
    

def plot_forecast(mean, ci_lower, ci_upper, title="Forecast with Confidence Interval"):
    """
    Plot forecast mean with confidence interval shading.
    
    mean: (horizon,) array of expected values
    ci_lower: (horizon,) array of lower CI bounds
    ci_upper: (horizon,) array of upper CI bounds
    """
    horizon = len(mean)
    x = range(horizon)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, mean, color="navy", label="Expected value", linewidth=2)
    plt.fill_between(x, ci_lower, ci_upper, color="skyblue", alpha=0.3, label="Confidence interval")
    
    plt.title(title, fontsize=14)
    plt.xlabel("Horizon", fontsize=12)
    plt.ylabel("Forecast", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_simulation_batch(returns, vols, percentiles=(5, 95)):
    
    _, horizon = returns.shape
    low_p, high_p = percentiles

    vol_med = jnp.percentile(vols, 50, axis=0)
    vol_low = jnp.percentile(vols, low_p, axis=0)
    vol_high = jnp.percentile(vols, high_p, axis=0)

    ret_med = jnp.percentile(returns, 50, axis=0)
    ret_low = jnp.percentile(returns, low_p, axis=0)
    ret_high = jnp.percentile(returns, high_p, axis=0)

    vol_med = jax.device_get(vol_med)
    vol_low = jax.device_get(vol_low)
    vol_high = jax.device_get(vol_high)

    ret_med = jax.device_get(ret_med)
    ret_low = jax.device_get(ret_low)
    ret_high = jax.device_get(ret_high)

    vols = jax.device_get(vols)
    returns = jax.device_get(returns)
    sample_idx = jax.device_get(sample_idx)
    x = jax.device_get(jnp.arange(horizon))

    fig, (ax_v, ax_r) = plt.subplots(1, 2, figsize=(12, 4))

    ax_v.fill_between(x, vol_low, vol_high, alpha=0.25)
    ax_v.plot(x, vol_med, linewidth=2, label="median")
    for i in sample_idx:
        ax_v.plot(x, vols[i], linewidth=0.8, alpha=0.6)
    ax_v.set_title("Volatility")
    ax_v.set_xlabel("Time")
    ax_v.set_ylabel("Volatility")
    ax_v.grid(True)

    ax_r.fill_between(x, ret_low, ret_high, alpha=0.25)
    ax_r.plot(x, ret_med, linewidth=2, label="median")
    for i in sample_idx:
        ax_r.plot(x, returns[i], linewidth=0.8, alpha=0.6)
    ax_r.set_title("Returns")
    ax_r.set_xlabel("Time")
    ax_r.set_ylabel("Return")
    ax_r.grid(True)

    fig.tight_layout()
    plt.show()

def plot_simulation_batch(returns_batch, title: str | None = None):
    """
    Plot cumulative-return paths and color them into `groups` by how far their final cumulative value is from the mean final value.
    The closest group (colors[0]) is plotted last so it appears on top.
    """
    colors = ['#6c2a6c', '#9f376e', '#bb4169', '#6c6b6c', '#d8575e', '#e88c6a', '#b9b3b9']
    groups = len(colors)

    rb = np.asarray(returns_batch, dtype=float)
    if rb.ndim == 1:
        rb = rb[np.newaxis, :]

    if rb.size == 0 or rb.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Simulated Return Forecast' if title is None else title)
        return fig, ax

    horizon = rb.shape[1]
    time_axis = np.arange(horizon)

    cumreturns = np.cumsum(rb, axis=1)
    n_sims = cumreturns.shape[0]

    final_vals = cumreturns[:, -1]
    mean_final = final_vals.mean()
    distances = np.abs(final_vals - mean_final)

    ranks = np.argsort(np.argsort(distances))
    group_idx = (ranks * groups) // max(n_sims, 1)
    group_idx = np.minimum(group_idx, groups - 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for g in range(groups - 1, -1, -1):
        inds = np.where(group_idx == g)[0]
        if inds.size == 0:
            continue
        if horizon == 1:
            ax.plot(np.zeros_like(inds), cumreturns[inds, 0], linestyle='', marker='o', color=colors[g], alpha=1, linewidth=1)
        else:
            segments = [np.column_stack((time_axis, cumreturns[i])) for i in inds]
            lc = LineCollection(segments, colors=colors[g], linewidths=1, alpha=1)
            ax.add_collection(lc)

    mean_path = cumreturns.mean(axis=0)
    ax.plot(time_axis, mean_path, 'k-', linewidth=1, label='Mean Path')

    ax.set_xlabel('Time Horizon')
    ax.set_ylabel('Returns')
    ax.set_title('Simulated Return Forecast' if title is None else title)

    y_min = min(cumreturns.min(), mean_path.min())
    y_max = max(cumreturns.max(), mean_path.max())
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    else:
        pad = (y_max - y_min) * 0.05
        y_min -= pad
        y_max += pad

    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_ylim(y_min, y_max)

    plt.show()
    return fig, ax