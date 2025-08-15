from fractrics._ts_components._core import ts_metadata
from tabulate import tabulate, SEPARATING_LINE
import numpy as np

#TODO: add standard errors/r., ...
# if hasattr() then concat
def summary(model: ts_metadata, latex = False, se_show = False, rse_show=False):
    tabformat = "latex" if latex else "simple"
    
    params_columns = [model.parameters.keys(), model.parameters.values()] 
    # list(model.parameters.items())
    params_head = [["", "Parameters"]]
    
    if hasattr(model, "standard_errors") and se_show:
        se_map = getattr(model, "standard_errors", {})
        params_head[0].append("Standard Errors")
        se_list = list(map(se_map.get, model.parameters.keys()))
        params_columns.append(se_list)
        
    if hasattr(model, "robust_standard_errors") and rse_show: 
        rse_map = getattr(model, "robust_standard_errors", {})
        params_head[0].append("Robust Standard Errors")
        rse_list = list(map(rse_map.get, model.parameters.keys()))
        params_columns.append(rse_list)
    
    params_columns = list(map(list, zip(*params_columns)))
    
    keys, values = list(model.hyperparameters.keys()), list(model.hyperparameters.values())
    hyperp_rows = list(sum(tuple(zip([keys], [values])), ()))
    
    content = []
    content += [["model:",  model.name]]
    content += [SEPARATING_LINE] + [["Hyperparameters", ""]] + [SEPARATING_LINE] + hyperp_rows
    content += [SEPARATING_LINE] + params_head + [SEPARATING_LINE] + params_columns
    content += [SEPARATING_LINE] + [["Likelihood:", model.optimization_info["negative_log_likelihood"]]]
    
    summary_table = tabulate(
        content,
        tablefmt=tabformat
    )
    
    print(summary_table)
    
    