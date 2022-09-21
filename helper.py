import os
import inspect
import shutil
import yaml
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from data import Gen

class BasicDataset(Dataset):

    def __init__(self, features, targets):

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):

        return len(self.features)
    
    def __getitem__(self, index):



        return self.features[index], self.targets[index]

def create_dataset(df, input_names, output_names, init = 1, length = 2):

    """
    takes DataFrame of ode states
    cuts each series in Dataframe up into samples with init and length
    then puts them into one torch Dataset as independent samples
    """

    # X = [1,4,9,16,25]

    # x0 = [1,4,0]
    # y0 = [4,9,16]

    # x1 = [4,9,0]
    # y1 = [9,16,25]

    input_names = list(input_names)
    output_names = list(output_names)

    # get some info about features
    num_inputs = len(input_names)
    num_outputs = len(output_names)

    # lists to store resulting samples
    trans_features = []
    trans_targets = []


    # loop over different series in DataFrame to handle them seperatly
    for _, series_data in df.groupby(level=0):

        # get targets and features from one series
        inputs = series_data[input_names].values
        outputs = series_data[output_names].values
        samples = len(inputs)- length

        # loop over single series 
        for index in range(samples):

            # cut together samples
            temp_features = np.zeros((length, num_inputs + num_outputs)) 
            temp_features[0:init, :num_outputs] = outputs[index:index + init]
            temp_features[:, num_outputs:] = inputs[index:index + length]
            temp_targets = outputs[index + 1:index + length + 1]

            # append to resulting lists
            trans_features.append(temp_features)
            trans_targets.append(temp_targets)
    
    # convert to numpy array (faster for torch)
    trans_features = np.array(trans_features)
    trans_targets = np.array(trans_targets)

    # finally give all samples to basic torch Dataset class
    data = BasicDataset(trans_features, trans_targets)

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

def save_dataset(df, config, path = "data/"):
    name = config["name"]
    if check_overwrite(name, path):
        savepath = f"{path}{name}"
        try:
            shutil.rmtree(savepath)
        except FileNotFoundError:
            pass
        print(savepath)
        os.makedirs(savepath, exist_ok=True)
        df.to_csv(f"{savepath}/{name}.csv")

        with open(f'{savepath}/{name}.json', 'w') as fp:
            json.dump(config, fp, indent=6)

def load_dataset(name, path = "data/"):
    savepath = f"{path}{name}"
    df = pd.read_csv(f"{savepath}/{name}.csv", index_col=[0, 1])
    with open(f'{savepath}/{name}.json', 'r') as stream:
        config = json.load(stream)
    return df, config

def gen_step(when, height, length):
    """
    create step impulse at certain (percent of total length [length]) points [when]
    and height [height]
    """

    out = []
    for w, h  in zip(when, height):
        data = np.zeros(length)
        for i in range(len(w)):
            pos = int(w[i]*length)
            data[pos:] = h[i]
        out.append(list(data))

    return out

# inputs: jump points, 
def gen_input(config):

    mem = {}

    samples = config["samples"]
    for inp, val in config["inputs"].items():
        mem[inp] = []
        for type, desc in val["types"].items():

            # step function
            if type == "steps":
                data = gen_step(desc["when"], desc["height"], samples)
                if len(np.shape(data)) == 2:
                    mem[inp] += data
                else:
                    mem[inp].append(data)

            # custom functions just takes lists given in custom
            if type == "custom":
                mem[inp] += desc

    # get all combinations of input variations
    comb = list(itertools.product(*mem.values()))
    df = pd.DataFrame()
    for idx, sample in enumerate(comb):
        index = [(idx, i) for i in range(samples)]
        index = pd.MultiIndex.from_tuples(index, names=["series", "index"])
        out_df = pd.DataFrame(np.array(sample).T, index= index, columns=config["inputs"].keys())
        df = pd.concat([df,out_df])

    return df

def gen_data(config, func):
    """
    generate ode results from inputs and other config options set in config dict, and returns it as Dataframe
    """

    # generate input dataframe and read constants, need to change to work with multiple constants
    input_df = gen_input(config)
    constants = tuple(config["constants"].values())

    # init counter and resulting Dataframe
    counter = 0
    result = pd.DataFrame()

    # loop over "series" index
    for _, data in input_df.groupby(level=0):
        # loop over different initial conditions given
        for init in config["init"]:

            # transform "inputs" array to fit in ode generator
            inputs = data[config["inputs"].keys()].T.values.tolist()
            inputs = np.array(inputs).T

            # calculate ode results for one current series
            x = Gen(func,(inputs,), constants, init,config["timestep"],config["samples"])
            x.generate()
            x.transform()

            # create new multiindex for dataframe
            index = [(counter, i) for i in range(config["samples"])]
            index = pd.MultiIndex.from_tuples(index, names=["series", "index"])

            # fill dataframe with new data and concat it with resulting dataframe to stack them above each other
            df = pd.DataFrame(np.array(x.X), index= index, columns=config["outputs"] + list(config["inputs"].keys()))
            result = pd.concat([result,df])

            counter += 1

    return result

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

    config = {"length": length, "features":features, "targets":targets}

    save_dataset(raw, config, "testing")

    # plt.scatter(t, data[0][0])
    # plt.scatter(t, data[0][1])
    # plt.show()
