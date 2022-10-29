from torch import nn
# NN architectures
class TwoLayers(nn.Module):
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
