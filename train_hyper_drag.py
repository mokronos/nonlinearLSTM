import os
import json
import shutil
import random
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from helper import load_dataset, create_dataset, save_model, check_overwrite, split_sets
from train import train
from models import *
import itertools

# set random seeds
torch.manual_seed(3)
random.seed(10)

#################################################
# define experiment parameters

name = "drag_step"
df, config = load_dataset(name)


# define experiment identifiers
descripor = "wholeseries"
version = "1"
dataset_name = config["name"]
# create full name for folder containing experiment
name = f"{dataset_name}_{descripor}_{version}"

# define dict with config info to store in json
experiment_config = {
        "name": name,
        "dataset_name" : dataset_name,
        "train_val_test_ratio" : [0.6, 0.2, 0.2],
        "architechture" : "NeuralNetwork",
        "epochs" : 200,
        "context_length": 1,
        "prediction_length": config["samples"] - 1,
        }

# define experiment parameters (gets added to experiment_config later)
lrs = [0.003, 0.0003, 0.0001]
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

# define device for compute
global device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# train/val/test ratio
ratio = experiment_config["train_val_test_ratio"]
num_series = len(df.groupby(level=0))

# get randomized indices split by ratio
train_idx, val_idx, test_idx = split_sets(ratio, num_series)

# split df into train/val/test df's with indices
df_train = df.loc[train_idx]
df_val = df.loc[val_idx]
df_test = df.loc[test_idx]

# print(df_train)
# print(df_val)
# print(df_test)

# save test set indices in config to test model on the correct data later
experiment_config["train_idx"] = train_idx
experiment_config["val_idx"] = val_idx
experiment_config["test_idx"] = test_idx
# memory for training and test loss for each hyperparameter sample
train_dict = {}
val_dict = {}

for params in hyper:

    # do validation set, do hyper search on val set
    # get best model and best epoch
    batch_size = params["batch_size"]

    ds_train = create_dataset(df_train, config["inputs"], config["outputs"], experiment_config["context_length"], experiment_config["prediction_length"])
    ds_val = create_dataset(df_test, config["inputs"], config["outputs"], experiment_config["context_length"], experiment_config["prediction_length"])

    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    # number of features
    input_size = len(config["inputs"]) + len(config["outputs"])
    # whatever is good? to be determined
    hidden_size = 500
    # number of outputs
    output_size = len(config["outputs"])

    model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
    print(model)
    epochs = experiment_config["epochs"]
    lr = params["lr"]
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# some lists for results
    train_loss = []
    val_loss = [np.nan]
    # best_state = deepcopy(model.state_dict())

    for t in range(epochs):

        if t%10==0:
            print(f"Epoch {t+1}\n-------------------------------")
        train_loss.append(train(train_dataloader, model, loss_fn, optimizer, device))

        # test model on validation set
        loss, _ = test(val_dataloader, model, loss_fn, device)
        # store loss and parameters of current model to later retrieve best model
        val_loss.append(loss)
        # state_dicts.append(state)


#################################################
# save model
    
    model_config = experiment_config.copy()
    model_config["name"] = f"{model_config['name']}_lr{lr}bs{batch_size}"
    model_config.update(params)

    save_model(savepath, model, model_config)

    print(f"saved {model_config['name']}!")

#################################################
# plot results

    train_dict[f"lr{lr}bs{batch_size}"] = train_loss
    val_dict[f"lr{lr}bs{batch_size}"] = val_loss

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(["train_loss", "val_loss"])

    plt.savefig(f"{savepath}/{model_config['name']}.pdf")
    plt.close()
    plt.clf()


# get best model, by checking model with min loss of val_dict
min_loss = 99999
min_epoch = -1
min_model_name = ""
for desc, loss in val_dict.items():
    m = min(loss[1:])
    if m < min_loss:
        min_loss = m
        min_epoch = int(np.argmin(loss[1:]))
        min_model_name = desc

best_model = {
        "best_model_name": min_model_name,
        "min_loss": min_loss,
        "epoch": min_epoch,
        }
print(best_model)
with open(f'{savepath}/best_model.json', 'w') as fp:
    json.dump(best_model, fp, indent=6)
print("best model:")
print(f"{min_model_name} with loss {min_loss} at epoch {min_epoch}")
plt.plot(np.array(list(val_dict.values())).T)
plt.legend(list(val_dict.keys()))
plt.savefig(f"{savepath}/val_loss_comparison.pdf")
plt.close()
plt.clf()
