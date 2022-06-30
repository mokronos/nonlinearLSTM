from ode import Gen

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable 


# X = np.array([[[1],[2],[3],[4]]])
# y = np.array([[[2],[3],[4],[5]]])

max = 20
data = np.array(range(max))

# make data into shape (samples, timeseries_length, num_features)
data = np.reshape(data, (1,max,1))
X = data[:,:-1,:]
y = data[:,1:,:]

train_test_split = 0.8
cutoff = int(data.shape[1] * (train_test_split))
# with more data there will probably be a train test split of the sequences, not the samples in the sequence
train_X, train_y = X[:,:cutoff], y[:,:cutoff]
test_X, test_y = X[:,cutoff:], y[:,cutoff:]

class timeseries(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

batch_size = 1

training_data = timeseries(train_X, train_y)
test_data = timeseries(test_X, test_y)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in train_dataloader:
    print(X)
    print(y)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_size = 10
        self.batch_size = 1
        self.lstm = nn.LSTM(1,self.hidden_size, batch_first=True)
        # self.ch = (torch.zeros(1,self.batch_size, self.hidden_size, device=device), torch.zeros(1,self.batch_size, self.hidden_size, device=device))
        self.fc = nn.Linear(self.hidden_size,128)
        self.fc1 = nn.Linear(128,1)
        self.relu = nn.ReLU()

    def forward(self, x):

        output, _ = self.lstm(x)
        output = self.relu(output)
        output = self.fc(output)
        output = self.relu(output)
        output = self.fc1(output)
        return output

model = LSTMNet().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print(X)
            print(y)
            print(pred)
            p = pred.to("cpu")
            yp = y.to("cpu")
            plt.plot(p.flatten())
            plt.plot(yp.flatten())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 50000
for t in range(epochs):
    if t%10000 == 0:
        print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
test(test_dataloader, model, loss_fn)
print("Done!")
# plt.show()
