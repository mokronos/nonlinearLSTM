import os
import matplotlib.pyplot as plt
import pandas as pd
from helper import load_data, load_json
from vis_helper import vis_results

# define all names
# dataset name
dataset_name = "drag_mult_step"
# experiment name
descriptor = "wholeseries"
version = "1"
# create full name for folder containing experiment
experiment_name = f"{dataset_name}_{descriptor}_{version}"
suffix = "train"

# where to save figures to
savedir = "figures/"
savepath = f"{savedir}{experiment_name}"

# creates dir if it doesn't exist
os.makedirs(savepath, exist_ok=True)

# load all 3 sets
data_config = load_json(dataset_name, dataset_name)
df_train = load_data(dataset_name, "train")
df_val = load_data(dataset_name, "val")
df_test = load_data(dataset_name, "test")
results = load_data(experiment_name, f"prediction_{suffix}", path="results/")

# vis anything

results = results.loc[:4]
vis_results(results, data_config, savepath, suffix)
