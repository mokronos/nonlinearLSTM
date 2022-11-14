import os
import shutil
import json
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from train import test_loop, train_tune
from helper import check_overwrite, create_dataset, load_data, load_json, norm_name, save_model
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig
from functools import partial
from models import *

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
model_config = {
        "name": name,
        "dataset_name" : dataset_name,
        "epochs" : 100,
        "bs" : 4,
        "arch" : "TwoLayers",
        "context_length": 1,
        "prediction_length": data_config["samples"] - 1,
        "norm": True,
        }

def auto_search(model_config, num_samples = 10, max_num_epochs = 200):
    """
    use raytune to auto search best hyperparameters
    """

    # define search space
    config = {
        "h1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "h2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(train_tune, model_config = model_config)),
            resources={"cpu": 2, "gpu": 0}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=RunConfig(local_dir="./ray_results"),
    )

    results = tuner.fit()

    best_result = results.get_best_result("loss", "min", "last")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))

    # os.chdir("/code/nonlinearLSTM")
    # load dataset config
    dataset_name = model_config["dataset_name"]
    data_config = load_json(dataset_name, dataset_name)

    # create model
    # number of features
    input_size = len(data_config["inputs"]) + len(data_config["outputs"])
    # whatever is good? to be determined
    hidden_size1 = best_result.config["h1"]
    hidden_size2 = best_result.config["h2"]
    # number of outputs
    output_size = len(data_config["outputs"])

    best_trained_model = eval(model_config["arch"])(input_size, hidden_size1, hidden_size2, output_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    # load best checkpoint
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    model_state, _ = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    # load test data (or validation data, to check if correct)
    test_df = load_data(dataset_name, "test")
    if model_config["norm"]:
        input_names = [norm_name(x) for x in data_config["inputs"]]
        output_names = [norm_name(x) for x in data_config["outputs"]]
    else:
        input_names = data_config["inputs"]
        output_names = data_config["outputs"]
    test_ds = create_dataset(test_df, input_names, output_names, 1, data_config["samples"] - 1)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

    loss_fn = torch.nn.MSELoss()
    test_loss,_ = test_loop(test_dataloader , best_trained_model, loss_fn, device)
    print("Best trial test set loss: {}".format(test_loss))

if __name__ == "__main__":
    auto_search(model_config)
