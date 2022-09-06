import os
import inspect
import shutil
import yaml
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SequenceDataset(Dataset):

    def __init__(self, features, targets, init, length):

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.num_targets = targets.shape[1]
        self.num_features = features.shape[1] + self.num_targets
        self.init = init
        self.length = length

    def __len__(self):

        return len(self.targets) - self.length

    def __getitem__(self, index):


        # X = [1,4,9,16,25]

        # x0 = [1,4,0]
        # y0 = [4,9,16]

        # x1 = [4,9,0]
        # y1 = [9,16,25]

        features = torch.zeros((self.length, self.num_features), dtype=torch.float32)
        features[0:self.init, :self.num_targets] = self.targets[index:index + self.init]
        features[:, self.num_targets:] = self.features[index:index + self.length]
        targets = self.targets[index + 1:index + self.length + 1]

        return features, targets

def create_dataset(df, features, targets, init = 1, length = 2):

    targets = df[targets].values
    features = df[features].values
    data = SequenceDataset(features, targets, init, length)
    return data


def check_overwrite(name, path):
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                if name == entry.name:
                    feedback = input("Overwrite?(y/n):")
                    if feedback == "y":
                        print(f'Overwriting "{name}" ...')
                        return True
                    else:
                        return False
    return True

def save_dataset(df, config, f, name, path = "data/"):
    if check_overwrite(name, path):
        savepath = f"{path}{name}"
        shutil.rmtree(savepath)
        os.makedirs(savepath, exist_ok=True)
        df.to_csv(f"{savepath}/{name}.csv")

        with open(f'{savepath}/{name}.yaml', 'w') as fp:
            yaml.dump(config, fp, indent=2)
            yaml.dump(f, fp,  indent=2, default_style="|")

def foo(a,b):
    return a+b

if __name__ == "__main__":
    length = 10
    t = np.arange(0,length)
    x = np.ones(length)
    y = t**2 + x

    raw = pd.DataFrame({"t": t, "y": y, "x": x})

    features = ["x"]
    targets = ["y"]

    data = create_dataset(raw, features, targets, 2, 3)


    dl = DataLoader(data, batch_size=1)

    print(raw.head())

    f = inspect.getsource(foo)
    config = {"length": length, "features":features, "targets":targets}
    f = {"function": f}

    save_dataset(raw, config,f, "testing")

    # plt.scatter(t, data[0][0])
    # plt.scatter(t, data[0][1])
    # plt.show()
