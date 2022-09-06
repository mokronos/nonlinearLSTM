import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from helper import create_dataset

# set random seed
torch.manual_seed(3)

#################################################
# NN architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        x,_ = self.lstm2(x)
        x = self.fc(x)
        return x

# test function
def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    # print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    return test_loss
#################################################
# load training and test data

data_path = "data/"
ext = ".pkl"
name = "getriebe"
df = pd.read_pickle(f"{data_path}{name}{ext}")
print(df.head())
features = ["impulse"]
targets=["torque_in", "torque_out", "torque_in_d", "torque_out_d"]

samples = len(df)
ratio = 0.8
cutoff = int(samples*ratio)
batch_size = 1

df_test = df[cutoff+1]

ds_test = create_dataset(df_test, features, targets, 3, 100)

test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

#################################################
# create NN and train

global device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

# load model
model_dir = "models/"
version = "1"
model_ext = "pt"
model_path = f"{model_dir}{name}{version}{model_ext}"

# number of features
input_size = len(features) + len(targets)
# whatever is good? to be determined
hidden_size = 500
# number of outputs
output_size = len(targets)

model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
print(model)

loss_fn = nn.MSELoss()

model.load_state_dict(torch.load(model_path))
model.eval()


loss = test(test_dataloader, model, loss_fn)

print(loss)
