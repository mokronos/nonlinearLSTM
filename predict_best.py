import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from helper import create_dataset, create_multiindex, load_data, load_json, prepare_folder, save_data
from models import *
from train import test_loop

# set random seed
torch.manual_seed(3)

#################################################
# load dataset for test data and variable information
name = "drag_mult_step"
data_config = load_json(name, name)
df_train = load_data(name, "train")
df_val = load_data(name, "val")
df_test = load_data(name, "test")

# define experiment identifiers
descriptor = "wholeseries"
version = "1"
name = data_config["name"]
# create full name for folder containing experiment
experiment_name = f"{name}_{descriptor}_{version}"

# prepare folder for saving results
result_dir = "results/"
variation = "base"
prepare_folder(experiment_name, result_dir)
savepath = f"{result_dir}{experiment_name}"

# load json to figure out what model is the best one
model_dir = "models/"
loadpath = f"{model_dir}{experiment_name}"
with open(f"{loadpath}/best_model.json", 'r') as stream:
    best_model = json.load(stream)

# load best model config
full_model_name = f"{experiment_name}_{best_model['best_model_name']}"
with open(f"{loadpath}/{experiment_name}_{best_model['best_model_name']}.json", 'r') as stream:
    model_config = json.load(stream)

samples = data_config["samples"]
batch_size = model_config["bs"]

# define which column of data to train on depending on if normalization is used
if model_config["norm"]:
    input_names = [f"{x}_norm" for x in list(data_config["inputs"])]
    output_names = [f"{x}_norm" for x in list(data_config["outputs"])]
else:
    input_names = list(data_config["inputs"])
    output_names = list(data_config["outputs"])


#################################################
# create NN and train

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

# load model
model_ext = ".pt"
model_path = f"{model_dir}{experiment_name}/{full_model_name}{model_ext}"

# number of features
input_size = len(data_config["inputs"]) + len(data_config["outputs"])
# whatever is good? to be determined
hidden_size = 500
# number of outputs
output_size = len(data_config["outputs"])

model = eval(model_config["arch"])(input_size,hidden_size,output_size).to(device)
print(model)

loss_fn = torch.nn.MSELoss()

model.load_state_dict(torch.load(model_path))
model.eval()
data = {"train":df_train, "val": df_val,"test": df_test}

for desc, data in data.items():

    ds = create_dataset(data, input_names, output_names, 1, samples - 1)

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    predictions = []
    ground_truth = []
    for X,y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X).detach().cpu().numpy()
        gt = y.detach().cpu().numpy()
        predictions.append(pred)
        ground_truth.append(gt)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    predictions = predictions.reshape((-1,)+predictions.shape[-2:])
    ground_truth = ground_truth.reshape((-1,)+predictions.shape[-2:])


    results = create_multiindex(predictions, ground_truth, data_config)

    save_data(results, experiment_name, f"prediction_{desc}", path=result_dir)

    loss, _ = test_loop(dataloader, model, loss_fn, device)

    print(f"loss on {desc}set: {loss}")
