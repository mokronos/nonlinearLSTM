import os
import matplotlib.pyplot as plt
import pandas as pd
from helper import load_data, load_json

# define all names
# dataset name
dataset_name = "pend_mult"
# experiment name
descriptor = "wholeseries"
version = "1"
# create full name for folder containing experiment
experiment_name = f"{dataset_name}_{descriptor}_{version}"
suffix = "val"

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

def norm_name(name):

    name_normed = f"{name}_norm"
    return name_normed

def pred_name(name):
    name_normed = f"{name}_pred"
    return name_normed

def gt_name(name):
    name_normed = f"{name}_gt"
    return name_normed

def error_name(name):
    name_normed = f"{name}_error"
    return name_normed

results = results.loc[:4]
indices = results.index.unique(level="series")
c = 0
for i in indices:


    data = results.xs(i)
    data.plot()

    for var in data_config["outputs"]:
        data[f"{var}_error"] =(data[gt_name(var)] - data[pred_name(var)])**2

    pred = data[[pred_name(x) for x in data_config["outputs"]]]
    gt = data[[gt_name(x) for x in data_config["outputs"]]]
    error = data[[error_name(x) for x in data_config["outputs"]]]

    fig, ax = plt.subplots(2, constrained_layout = True)
    ax[0].plot(pred, label=[f"pred_{x}" for x in data_config["outputs"]])
    ax[0].plot(gt, "--", label=[f"gt_{x}" for x in data_config["outputs"]])
    ax[0].set_ylabel(f"{data_config['output_labels'][0]} in {data_config['output_units'][0]}")
    ax[0].set_xlabel("time in 0.01s steps")
    ax[0].legend()
    ax[1].plot(error, label=[f"squ_error_{x}" for x in data_config["outputs"]])
    ax[1].set_ylabel(r"$m^2/s^2$")
    ax[1].set_xlabel("time in 0.01s steps")
    ax[1].legend()
    plt.savefig(f"{savepath}/{suffix}{c}.pdf")
    c+=1
