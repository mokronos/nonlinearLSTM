import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
# load training and test data

savepath = "data/"

df = pd.read_pickle(savepath + "pendulum.pkl")

X = np.expand_dims(df[["dangle", "angle", "force_input"]][:-1], axis=0)
X[0,1:,:2] = 0
y = np.expand_dims(df[["dangle", "angle"]][1:], axis=0)

# cutoff = int(X.shape[1] * 2/3 )
cutoff = 200
X_train, y_train = X[:,:cutoff], y[:,:cutoff]
# X_test, y_test = X[:,cutoff:], y[:,cutoff:]
X_test, y_test = X[:,:cutoff], y[:,:cutoff]
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

epochs = 500
lr = 1e-4
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

# for param in model.parameters():
#     print(param.data)

plt.plot(loss_vals)
plt.title("loss")

df[["pred_dangle", "pred_angle"]] = np.nan
# df.loc[cutoff+1:, ["pred_dangle", "pred_angle"]] = test_results[0][0].numpy()
df.loc[:cutoff-1, ["pred_dangle", "pred_angle"]] = test_results[0][0].numpy()
df["rmse_angle"] = ((df.angle - df.pred_angle)**2).mean() ** 0.5
df["error_angle"] = df.angle - df.pred_angle
df["rmse_dangle"] = ((df.dangle - df.pred_dangle)**2).mean() ** 0.5
df["error_dangle"] = df.dangle - df.pred_dangle
print(df.head())
_ ,ax = plt.subplots(2)
# df[["dangle","pred_dangle"]][cutoff:].plot(ax = ax[0])
df[["dangle","pred_dangle","error_dangle"]][:cutoff].plot(ax = ax[0], color=["blue", "orange", 'mediumvioletred'])
ax[0].set(ylabel="rad/s")
ax[0].set_title("dangle")
# df[["angle","pred_angle"]][cutoff:].plot(ax = ax[1])
df[["angle","pred_angle", "error_angle"]][:cutoff].plot(ax = ax[1], color=["blue", "orange", 'mediumvioletred'])
ax[1].set(ylabel="rad")
ax[1].set_title("angle")
for ax in ax.flat:
    ax.set(xlabel="timestep")
    ax.grid()
plt.show()
