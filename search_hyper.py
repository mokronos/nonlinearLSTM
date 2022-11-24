import os
import time
import sys
import json
import torch
import random
import itertools
import numpy as np
from train import train
from helper import check_overwrite, create_summary, load_json, save_model
from predict_best import predict_best_model
import pandas as pd

# set random seeds
torch.manual_seed(3)
random.seed(10)

# dataset to load
# dataset_name = "drag_simple_steps"
# dataset_name = "drag_complex"
# dataset_name = "pend_simple"
# dataset_name = "pend_complex"
# dataset_name = "drag_complex_var"
dataset_name = "pend_simple_var"

# define experiment identifiers
descriptor = "alpha"
version = "4"
# create full name for folder containing experiment
experiment_name = f"{dataset_name}_{descriptor}_{version}"

# load dataset_config to get length of series to define length of prediction
data_config = load_json(dataset_name, dataset_name)

# define dict with config info to store in json, need to make batch size more automatic
experiment_config = {
        "name": experiment_name,
        "dataset_name" : dataset_name,
        "arch": "TwoLayers",
        "epochs" : 5000,
        "bs" : 9,
        "context_length": 1,
        "prediction_length": data_config["samples"] - 1,
        "norm": True,
        }

# define experiment parameters (gets added to experiment_config later)
# learning rate
# lr = [0.01,0.005,0.001,0.0005,0.0001]
lr = [0.001]
arch = ["TwoLayers"]
# arch = ["OneLayers", "TwoLayers", "ThreeLayers", "FourLayers", "FiveLayers"]
# nodes = [32,64,128,256,512]
nodes = [64]

# descriptors for the hyperparameters
# add to model_config later
hyper_desc = ["lr", "arch", "nodes"]


# get all combinations of params and put into dicts with descripors as keys
hyper = list(itertools.product(lr, arch, nodes))
hyper = [{desc: par for desc, par in zip(hyper_desc, params)} for params in hyper]


# create folder for experiment
path = "models/"
savepath = f"{path}{experiment_config['name']}"

# check overwrite, just to remind yourself to check everything twice
if check_overwrite(experiment_config["name"], path):
    pass
else:
    sys.exit()

os.makedirs(savepath, exist_ok=True)


for params in hyper:

    # create model config by expanding experiment config with hyperparams
    param_desc = "".join([f"{desc}{par}" for desc, par in params.items()])
    model_config = experiment_config.copy()
    model_config["name"] = f"{model_config['name']}_{param_desc}"
    model_config["hyper_desc"] = hyper_desc
    model_config.update(params)

    # time
    start_time = time.time()

    # train model with current hyperparams
    best_state, train_loss_hist, val_loss_hist = train(model_config)

    # time
    end_time = time.time()
    # print time in minutes:seconds
    print(f"Training took: {int((end_time - start_time) // 60)} min and {int((end_time - start_time) % 60)} sec")

    # save model
    save_model(savepath, best_state, model_config)
    print(f"saved {model_config['name']}!")

    
    # save train and val loss to plot comparison
    if np.isnan(val_loss_hist).all():
        best_val_loss = np.nan
        best_epoch = 0
    else:
        best_val_loss= np.nanmin(val_loss_hist)
        best_epoch = np.nanargmin(val_loss_hist)

    # create dict for json
    best_stats = {
            "desc": model_config["name"],
            "best_val_loss": best_val_loss,
            "best_epoch": int(best_epoch),
            }

    # save best val loss and best epoch to json
    with open(f"{savepath}/{model_config['name']}_best.json", "w") as fp:
        json.dump(best_stats, fp, indent=6)

    # save train/val plots for each parameter combination
    loss_df = pd.DataFrame({"train_loss": train_loss_hist, "val_loss": val_loss_hist})
    loss_df.to_csv(f"{savepath}/{model_config['name']}_loss.csv")

# create/update summary
create_summary(experiment_name)

# predict with best model and save results
predict_best_model(dataset_name, descriptor, version)
