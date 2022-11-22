import torch
import os
import numpy as np
from copy import deepcopy
from torch.utils.data.dataloader import DataLoader
from models import *
from helper import create_dataset, load_data, load_json, norm_name
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint


def train(model_config):

    """
    trains one model described by model_config with data
    """
    # fix random seed in here, doesn't work in other file
    torch.manual_seed(3)
    
    # load data
    dataset_name = model_config["dataset_name"]
    data_config = load_json(dataset_name, dataset_name)
    train_df = load_data(dataset_name, "train")
    val_df = load_data(dataset_name, "val")

    # define which column of data to train on depending on if normalization is used
    if model_config["norm"]:
        input_names = [norm_name(x) for x in data_config["inputs"]]
        output_names = [norm_name(x) for x in data_config["outputs"]]
    else:
        input_names = data_config["inputs"]
        output_names = data_config["outputs"]

    # create train and validation dataset
    ds_train = create_dataset(train_df, input_names, output_names, model_config["context_length"], model_config["prediction_length"])
    ds_val = create_dataset(val_df, input_names, output_names, model_config["context_length"], model_config["prediction_length"])

    batch_size = model_config["bs"]
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    # create model
    # number of features
    input_size = len(data_config["inputs"]) + len(data_config["outputs"])

    # define hidden nodes, pack in list so network can be variable depth (rest gets ignored)
    # if nodes not None, make all layers have same number of nodes
    nodes = model_config["nodes"]
    if nodes:
        h1 = h2 = h3 = h4 = h5 = nodes
    else:
        h1 = model_config["h1"]
        h2 = model_config["h2"]
        h3 = model_config["h3"]
        h4 = model_config["h4"]
        h5 = model_config["h5"]
    hidden_nodes = [h1, h2, h3, h4, h5]

    # number of outputs
    output_size = len(data_config["outputs"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = eval(model_config["arch"])(input_size, output_size, *hidden_nodes).to(device)
    print(model)

    # loop over epochs
    epochs = model_config["epochs"]
    lr = model_config["lr"]
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # some lists for results
    train_loss_hist = []
    # don't know first validation loss, as there is one optimization step before checking validation loss
    val_loss_hist = [np.nan]
    best_state = deepcopy(model.state_dict())
    best_loss, _ = test_loop(val_dataloader, model, loss_fn, device)

    for epoch in range(epochs):

        if epoch%10==0:
            print(f"Epoch {epoch}\n-------------------------------")
        train_loss_hist.append(train_loop(train_dataloader, model, loss_fn, optimizer, device))

        # test model on validation set
        val_loss, model_state = test_loop(val_dataloader, model, loss_fn, device)

        # save best model by comparing current loss to best loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model_state

        # store loss and parameters of current model to later retrieve best model
        val_loss_hist.append(val_loss)
    
    # append nan value to train loss to have same length as val loss
    train_loss_hist.append(np.nan)

    return best_state, train_loss_hist, val_loss_hist


# define training and test loops (pretty much default from pytorch quickstart)
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    acc_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # add up loss
        acc_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return average loss over batches
    avg_loss = acc_loss/len(dataloader)
    return avg_loss


def test_loop(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # add up loss
            test_loss += loss_fn(pred, y).item()

    # return average loss over batches
    avg_loss = test_loss / num_batches
    model_state = deepcopy(model.state_dict())
    return avg_loss, model_state

def train_tune(config, model_config):

    """
    trains one model described by model_config with data, adjusted for raytune
    """
    # fix random seed in here, doesn't work in other file
    torch.manual_seed(3)
    
    try:
        os.chdir("/home/mokronos/code/nonlinearLSTM")
    except:
        pass

    # load data
    dataset_name = model_config["dataset_name"]
    data_config = load_json(dataset_name, dataset_name)
    train_df = load_data(dataset_name, "train")
    val_df = load_data(dataset_name, "val")

    # define which column of data to train on depending on if normalization is used
    if model_config["norm"]:
        input_names = [norm_name(x) for x in data_config["inputs"]]
        output_names = [norm_name(x) for x in data_config["outputs"]]
    else:
        input_names = data_config["inputs"]
        output_names = data_config["outputs"]

    # create train and validation dataset
    ds_train = create_dataset(train_df, input_names, output_names, model_config["context_length"], model_config["prediction_length"])
    ds_val = create_dataset(val_df, input_names, output_names, model_config["context_length"], model_config["prediction_length"])

    batch_size = model_config["bs"]
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    # create model
    # number of features
    input_size = len(data_config["inputs"]) + len(data_config["outputs"])
    # whatever is good? to be determined
    hidden_size1 = config["h1"]
    hidden_size2 = config["h2"]
    # number of outputs
    output_size = len(data_config["outputs"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = eval(model_config["arch"])(input_size, hidden_size1, hidden_size2, output_size).to(device)
    # print(model)

    # loop over epochs
    epochs = model_config["epochs"]
    lr = config["lr"]
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # load checkpoint
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(epochs):

        # if epoch%10==0:
        #     print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)

        # test model on validation set
        val_loss, _ = test_loop(val_dataloader, model, loss_fn, device)

        os.makedirs("ray_checkpoint", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "ray_checkpoint/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("ray_checkpoint")
        session.report({"loss": val_loss}, checkpoint=checkpoint)
