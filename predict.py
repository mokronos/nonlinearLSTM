import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from helper import create_dataset, load_dataset

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

name = "pendulum_2init0force"
df, config = load_dataset(name)
ratio = 0.5
samples = config["samples"]
num_series = len(df.groupby(level=0))
cutoff = int(num_series*ratio)
batch_size = 1
cutoff = 1
df_test = df.loc[[cutoff]]
print(df_test)

ds_test = create_dataset(df_test, config["inputs"], config["outputs"], 3, samples - 1)

test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

#################################################
# create NN and train

global device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

# load model
model_dir = "models/"
dataset_name = config["name"]
model_name = "simple"
version = "1"
model_ext = ".pt"
model_path = f"{model_dir}{dataset_name}_{model_name}{version}{model_ext}"

# number of features
input_size = len(config["inputs"]) + len(config["outputs"])
# whatever is good? to be determined
hidden_size = 500
# number of outputs
output_size = len(config["outputs"])

model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
print(model)

loss_fn = nn.MSELoss()

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
ax[0].set_ylabel("radians")
ax[0].set_xlabel("time in 0.01s steps")
ax[0].legend()
ax[1].plot((ground_truth[0][0] - predictions[0][0])**2, label=[f"squ_error_{x}" for x in config["outputs"]])
ax[1].set_ylabel("radians^2")
ax[1].set_xlabel("time in 0.01s steps")
ax[1].legend()
plt.show()
loss = test(test_dataloader, model, loss_fn)

print(loss)
