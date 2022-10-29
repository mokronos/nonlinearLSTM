import os
import shutil
import json
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from train import train
from helper import check_overwrite, load_dataset, save_model, split_sets

# set random seeds
torch.manual_seed(3)
random.seed(10)

# load dataset
name = "drag_step"
df, config = load_dataset(name)


# define experiment identifiers
descripor = "wholeseries"
version = "3"
dataset_name = config["name"]
# create full name for folder containing experiment
name = f"{dataset_name}_{descripor}_{version}"


# define dict with config info to store in json
experiment_config = {
        "name": name,
        "dataset_name" : dataset_name,
        "train_val_test_ratio" : [0.6, 0.2, 0.2],
        "architechture" : "NeuralNetwork",
        "epochs" : 10,
        "context_length": 1,
        "prediction_length": config["samples"] - 1,
        }

# define experiment parameters (gets added to experiment_config later)
lrs = [0.0001]
# lrs = [0.003, 0.0003, 0.0001]
batch_size = [1]
hyper_desc = ["lr", "batch_size"]

# get all combinations of params and put into dicts with descripors as keys
hyper = list(itertools.product(lrs, batch_size))
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

# train/val/test ratio
ratio = experiment_config["train_val_test_ratio"]
num_series = len(df.groupby(level=0))

# get randomized indices split by ratio
train_idx, val_idx, test_idx = split_sets(ratio, num_series)

# split df into train/val/test df's with indices
df_train = df.loc[train_idx]
df_val = df.loc[val_idx]

# print(df_train)
# print(df_val)

# save test set indices in config to test model on the correct data later
experiment_config["train_idx"] = train_idx
experiment_config["val_idx"] = val_idx
experiment_config["test_idx"] = test_idx

train_dict = {}
val_dict = {}

best_val_losses = {}
best_epochs = {}

for params in hyper:

    lr = params["lr"]
    batch_size = params["batch_size"]
    model_config = experiment_config.copy()
    model_config["name"] = f"{model_config['name']}_lr{lr}bs{batch_size}"
    model_config.update(params)

    best_state, train_loss_hist, val_loss_hist = train(df_train, df_val, config, model_config)

    # save model
    save_model(savepath, best_state, model_config)
    print(f"saved {model_config['name']}!")
    
    # save train and val loss to plot comparison
    train_dict[f"lr{lr}bs{batch_size}"] = train_loss_hist
    val_dict[f"lr{lr}bs{batch_size}"] = val_loss_hist
    best_val_losses[f"lr{lr}bs{batch_size}"] = np.nanmin(val_loss_hist)
    best_epochs[f"lr{lr}bs{batch_size}"] = np.nanargmin(val_loss_hist)


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
plt.legend(list(val_dict.keys()))
plt.savefig(f"{savepath}/val_loss_comparison.pdf")
plt.close()
plt.clf()
plt.plot(np.array(list(train_dict.values())).T)
plt.legend(list(val_dict.keys()))
plt.savefig(f"{savepath}/train_loss_comparison.pdf")
plt.close()
plt.clf()
