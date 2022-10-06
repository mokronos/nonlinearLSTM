import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from helper import load_dataset, create_dataset

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

#################################################
# define training and test loops (pretty much default from pytorch quickstart)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    acc_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        acc_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return acc_loss/len(dataloader)


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

name = "thermal_simple"
df, config = load_dataset(name)

# train/test ratio
ratio = 0.8
num_series = len(df.groupby(level=0))
cutoff = int(num_series*ratio)
batch_size = 100

df_train = df.loc[:cutoff]
df_test = df.loc[cutoff:]

ds_train = create_dataset(df_train, config["inputs"], config["outputs"], 3, 20)
ds_test = create_dataset(df_test, config["inputs"], config["outputs"], 3, 20)

train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

#################################################
# create NN and train

global device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

# number of features
input_size = len(config["inputs"]) + len(config["outputs"])
# whatever is good? to be determined
hidden_size = 500
# number of outputs
output_size = len(config["outputs"])

model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
print(model)

epochs = 20
lr = 3e-4
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# some lists for results
train_loss = []
test_loss = []

for t in range(epochs):
    
    if t%1==0:
        print(f"Epoch {t+1}\n-------------------------------")
    train_loss.append(train(train_dataloader, model, loss_fn, optimizer))

    test_loss.append(test(test_dataloader, model, loss_fn))

print("Done!")

#################################################
# save model

model_dir = "models/"
dataset_name = config["name"]
model_name = "testing"
version = "1"
model_ext = ".pt"
model_path = f"{model_dir}{dataset_name}_{model_name}{version}{model_ext}"

torch.save(model.state_dict(), model_path)


#################################################
# plot results

plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(["train_loss", "test_loss"])

plt.show()
