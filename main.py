import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from data import Gen

import matplotlib.pyplot as plt

# set random seed
torch.manual_seed(3)

#################################################
# Dataset custom class 
class BasicDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]


# NN architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x

#################################################
# define training and test loops (pretty much default from pytorch quickstart)

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
        loss_vals.append(loss.item())

        # for weight in model.named_parameters():
        #     print(f"{weight}")

        # print(f"loss: {loss.item()}")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # test_results.append(pred.cpu())
            # change this, its ugly
            test_results.append(pred.cpu())
            # print(X,y)
            # print(pred)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")


#################################################
# create training and test data


# X 1s --> y time dependent but linear
# y = np.array(list(range(1,10)))
# X = np.array([1]*len(y))
# y = y.reshape((1,len(y),1))
# X = X.reshape((1,len(X),1))

# X = np.array(list(range(1,10)))
# X = X.reshape((1,len(X),1))
# y = X

def func(y,t,u,g):
    l = 1

    res = [y[1], -(g/l) * np.sin(y[0]) + u[0] + u[1]]
    return res

# [angle'_init, angle_init]
y0 = [0, 0.1]
samples = 500
dt = 0.01

# if parameters should vary in time, give them as lists like this for [[par1(t=0),par2(t=0)], [par1(t=1),par2(t=1)], ... ]; just append them as list, then transpose
# if single constant, just give them as one value
u = []
u.append([0]*samples)
u.append([1]*samples)
u = np.array(u)
u = u.T
g = 9.81

# give input, then parameters (both as tuples); inputs = things the RNN/model gets as input as well, parameters = things the model is supposed to learn (potential changes in the system)
x = Gen(func,(u,),(g,),y0,dt,samples)
x.generate()
x.transform()


X = np.expand_dims(x.X, axis=0)
y = np.expand_dims(x.y, axis=0)

cutoff = int(X.shape[1] * 2/3 )
X_train, y_train = X[:,:cutoff], y[:,:cutoff]
X_test, y_test = X[:,cutoff:], y[:,cutoff:]
# X_train, y_train = X[:,:6], y[:,:6]
# X_test, y_test = X[:,:6], y[:,:6]
# print(X_train)

training_data = BasicDataset(X_train, y_train)
test_data = BasicDataset(X_test, y_test)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#################################################
# create NN and train

global device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

# number of features
input_size = X_train.shape[2]
# whatever is good? to be determined
hidden_size = 20
# number of outputs
output_size = y_train.shape[2]

model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
print(model)

epochs = 1000
lr = 1e-1
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# some lists for results
global loss_vals
loss_vals = []
global test_results
test_results = []

for t in range(epochs):
    
    if t%100==0:
        print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)

test(test_dataloader, model, loss_fn)

print("Done!")

#################################################
# plot results

_,ax = plt.subplots(3)
ax[0].plot(loss_vals)
ax[1].plot(y_test[0])
ax[2].plot(test_results[0][0])
plt.show()
