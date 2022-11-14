import os
import shutil
import json
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from train import train
from helper import check_overwrite, load_data, load_json, save_model

# set random seeds
torch.manual_seed(3)
random.seed(10)

# load dataset
dataset_name = "drag_mult_step"

# define experiment identifiers
descripor = "test"
version = "1"
# create full name for folder containing experiment
name = f"{dataset_name}_{descripor}_{version}"

# load dataset_config to get length of series to define length of prediction
data_config = load_json(dataset_name, dataset_name)

# define dict with config info to store in json
experiment_config = {
        "name": name,
        "dataset_name" : dataset_name,
        "epochs" : 2,
        "context_length": 1,
        "prediction_length": data_config["samples"] - 1,
        "norm": True,
        "h1": 64,
        "h2": 64,
        }

# define experiment parameters (gets added to experiment_config later)
# learning rate
lrs = [ 0.003, 0.0001]
# lrs = [0.003, 0.0003, 0.0001]
# batch_size
bs = [4]
# define different architechtures to test
arch = ["TwoLayers"]

# descriptors for the hyperparameters
hyper_desc = ["lr", "bs", "arch"]

# get all combinations of params and put into dicts with descripors as keys
hyper = list(itertools.product(lrs, bs, arch))
hyper = [{desc: par for desc, par in zip(hyper_desc, params)} for params in hyper]

# create folders
# check if experiment already exist,
# if yes, ask if overwrite
# else, quit
path = "models/"
savepath = f"{path}{experiment_config['name']}"
if check_overwrite(experiment_config["name"], path):
    try:
        shutil.rmtree(savepath)
    except FileNotFoundError:
        pass
    print(savepath)
    os.makedirs(savepath, exist_ok=True)
else:
    quit()


train_dict = {}
val_dict = {}

best_val_losses = {}
best_epochs = {}

for params in hyper:

    param_desc = "".join([f"{desc}{par}" for desc, par in params.items()])
    model_config = experiment_config.copy()
    model_config["name"] = f"{model_config['name']}_{param_desc}"
    model_config.update(params)

    best_state, train_loss_hist, val_loss_hist = train(model_config)

    # save model
    save_model(savepath, best_state, model_config)
    print(f"saved {model_config['name']}!")
    
    # save train and val loss to plot comparison
    train_dict[param_desc] = train_loss_hist
    val_dict[param_desc] = val_loss_hist
    best_val_losses[param_desc] = np.nanmin(val_loss_hist)
    best_epochs[param_desc] = np.nanargmin(val_loss_hist)

    # save train/val plots for each parameter combination
    plt.plot(train_loss_hist)
    plt.plot(val_loss_hist)
    plt.yscale("log")
    plt.legend(["train_loss", "val_loss"])

    plt.savefig(f"{savepath}/{model_config['name']}.pdf")
    plt.savefig(f"{savepath}/{model_config['name']}.png")
    plt.close()
    plt.clf()

best_desc = min(best_val_losses, key=best_val_losses.get)
min_loss = best_val_losses[best_desc]
best_epoch = int(best_epochs[best_desc])

best_model = {
        "best_model_name": best_desc,
        "min_loss": min_loss,
        "epoch": best_epoch,
        }

print(best_model)
with open(f'{savepath}/best_model.json', 'w') as fp:
    json.dump(best_model, fp, indent=6)
plt.plot(np.array(list(val_dict.values())).T)
plt.yscale("log")
plt.legend(list(val_dict.keys()))
plt.savefig(f"{savepath}/val_loss_comparison.pdf")
plt.savefig(f"{savepath}/val_loss_comparison.png")
plt.close()
plt.clf()
plt.plot(np.array(list(train_dict.values())).T)
plt.yscale("log")
plt.legend(list(val_dict.keys()))
plt.savefig(f"{savepath}/train_loss_comparison.pdf")
plt.savefig(f"{savepath}/train_loss_comparison.png")
plt.close()
plt.clf()

