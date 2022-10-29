import torch
import numpy as np
from copy import deepcopy
from torch.utils.data.dataloader import DataLoader
from models import *
from helper import create_dataset


def train(train_data, val_data, config, train_config):

    """
    trains one model described by train_config with data
    """

    # create train and validation dataset
    ds_train = create_dataset(train_data, config["inputs"], config["outputs"], train_config["context_length"], train_config["prediction_length"])
    ds_val = create_dataset(val_data, config["inputs"], config["outputs"], train_config["context_length"], train_config["prediction_length"])

    batch_size = train_config["batch_size"]
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    # create model
    # number of features
    input_size = len(config["inputs"]) + len(config["outputs"])
    # whatever is good? to be determined
    hidden_size = 500
    # number of outputs
    output_size = len(config["outputs"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
    print(model)

    # loop over epochs
    epochs = train_config["epochs"]
    lr = train_config["lr"]
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # some lists for results
    train_loss_hist = []
    # don't know first validation loss, as there is one optimization step before checking validation loss
    val_loss_hist = [np.nan]
    best_state = deepcopy(model.state_dict())

    for epoch in range(epochs):

        if epoch%10==0:
            print(f"Epoch {epoch}\n-------------------------------")
        train_loss_hist.append(train_loop(train_dataloader, model, loss_fn, optimizer, device))

        # test model on validation set
        loss, model_state = test_loop(val_dataloader, model, loss_fn, device)

        if loss < val_loss_hist[-1]:
            best_state = model_state
        # store loss and parameters of current model to later retrieve best model
        val_loss_hist.append(loss)
        # state_dicts.append(state)



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
