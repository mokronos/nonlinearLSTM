import os
import matplotlib.pyplot as plt
import pandas as pd
from helper import get_entries, load_data, load_json, load_result
from vis_helper import vis_data_1, vis_data_2, vis_results_1, vis_results_2

# define some global matplotlib params
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# define all names
# dataset name
# dataset_name = "pend_simple"
dataset_name = "pend_simple_var"
# dataset_name = "pend_complex"
# dataset_name = "drag_simple_steps"
# dataset_name = "drag_complex"
# dataset_name = "drag_complex_var"

# experiment name
descriptor = "alpha"
version = "4"
variation = "base"
# create full name for folder containing experiment
experiment_name = f"{dataset_name}_{descriptor}_{version}"
suffixes = ["train", "val", "test"] 

# where to save figures to
savedir = "figures/"
experiment_savepath = f"{savedir}{experiment_name}"
data_savepath = f"{savedir}{dataset_name}"

# creates dir if it doesn't exist
os.makedirs(experiment_savepath, exist_ok=True)
os.makedirs(data_savepath, exist_ok=True)

# load all 3 sets
data_config = load_json(dataset_name, dataset_name)
df_train = load_data(dataset_name, "train")
df_val = load_data(dataset_name, "val")
df_test = load_data(dataset_name, "test")

# load results for best model
results_train = load_result(experiment_name, variation, "train")
results_val = load_result(experiment_name, variation, "val")
results_test = load_result(experiment_name, variation, "test")


# df_train = get_entries(df_train, 10)
# df_val = get_entries(df_val, 10)
# df_test = get_entries(df_test, 10)

for suff in suffixes:
    if len(data_config["outputs"]) == 1:
        vis_data_1(eval(f"df_{suff}"), data_config, data_savepath, f"{dataset_name}_{suff}")
    else:
        vis_data_2(eval(f"df_{suff}"), data_config, data_savepath, f"{dataset_name}_{suff}")

if len(data_config["outputs"]) == 1:
    vis_results_1(results_test, data_config, experiment_savepath, f"{experiment_name}_results") 
else:
    vis_results_2(results_test, data_config, experiment_savepath, f"{experiment_name}_results") 
