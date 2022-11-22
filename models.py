from torch import nn
# NN architectures

class OneLayers(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1, *_):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.fc = nn.Linear(hidden_size1,output_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x

class TwoLayers(nn.Module):
    def __init__(self, input_size, output_size,  hidden_size1, hidden_size2, *_):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2,output_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        x,_ = self.lstm2(x)
        x = self.fc(x)
        return x

class ThreeLayers(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, hidden_size3, *_):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.fc = nn.Linear(hidden_size3,output_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        x,_ = self.lstm2(x)
        x,_ = self.lstm3(x)
        x = self.fc(x)
        return x

class FourLayers(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, *_):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_size3, hidden_size4, batch_first=True)
        self.fc = nn.Linear(hidden_size4,output_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        x,_ = self.lstm2(x)
        x,_ = self.lstm3(x)
        x,_ = self.lstm4(x)
        x = self.fc(x)
        return x

class FiveLayers(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, *_):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_size3, hidden_size4, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_size4, hidden_size5, batch_first=True)
        self.fc = nn.Linear(hidden_size5,output_size)

    def forward(self, x):
        x,_ = self.lstm(x)
        x,_ = self.lstm2(x)
        x,_ = self.lstm3(x)
        x,_ = self.lstm4(x)
        x,_ = self.lstm5(x)
        x = self.fc(x)
        return x
