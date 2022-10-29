import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from helper import create_dataset, load_dataset
from models import *
from train import test_loop

# set random seed
torch.manual_seed(3)

#################################################
# load dataset for test data and variable information
dataset_name = "drag_step"
df, config = load_dataset(dataset_name)

# define experiment identifiers
descripor = "wholeseries"
version = "3"
dataset_name = config["name"]
# create full name for folder containing experiment
experiment_name = f"{dataset_name}_{descripor}_{version}"

# load json to figure out what model is the best one
model_dir = "models/"
savepath = f"{model_dir}{experiment_name}"
with open(f"{savepath}/best_model.json", 'r') as stream:
    best_model = json.load(stream)

# load best model config
full_model_name = f"{experiment_name}_{best_model['best_model_name']}"
with open(f"{savepath}/{experiment_name}_{best_model['best_model_name']}.json", 'r') as stream:
    model_config = json.load(stream)

samples = config["samples"]
batch_size = 1
# df_test = df.loc[model_config["test_idx"]]
print(model_config["test_idx"])
df_test = df.loc[[0]]

ds_test = create_dataset(df_test, config["inputs"], config["outputs"], 1, samples - 1)

test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

#################################################
# create NN and train

global device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

# load model
model_ext = ".pt"
model_path = f"{model_dir}{experiment_name}/{full_model_name}{model_ext}"

# number of features
input_size = len(config["inputs"]) + len(config["outputs"])
# whatever is good? to be determined
hidden_size = 500
# number of outputs
output_size = len(config["outputs"])

model = eval(model_config["arch"])(input_size,hidden_size,output_size).to(device)
print(model)

loss_fn = torch.nn.MSELoss()

model.load_state_dict(torch.load(model_path))
model.eval()
predictions = []
ground_truth = []
for X,y in test_dataloader:
    X, y = X.to(device), y.to(device)
    pred = model(X).detach().cpu().numpy()
    predictions.append(pred)
    ground_truth.append(y.detach().cpu().numpy())
fig, ax = plt.subplots(2)
ax[0].plot(predictions[0][0], label=[f"pred_{x}" for x in config["outputs"]])
ax[0].plot(ground_truth[0][0], "--", label=[f"gt_{x}" for x in config["outputs"]])
ax[0].set_ylabel(r"m/s")
ax[0].set_xlabel("time in 0.01s steps")
ax[0].legend()
ax[1].plot((ground_truth[0][0] - predictions[0][0])**2, label=[f"squ_error_{x}" for x in config["outputs"]])
ax[1].set_ylabel(r"m^2/s^2")
ax[1].set_xlabel("time in 0.01s steps")
ax[1].legend()
plt.show()
loss, _ = test_loop(test_dataloader, model, loss_fn, device)

print(loss)
