import os
import glob
import random
import shutil
import json
import joblib
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import itertools
from data import Gen
from matplotlib import pyplot as plt

class BasicDataset(Dataset):

    def __init__(self, features, targets):

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):

       return len(self.features)

    def __getitem__(self, index):

       return self.features[index], self.targets[index]

def norm_name(name):
    return f"{name}_norm"

def pred_name(name):
    return f"{name}_pred"

def gt_name(name):
    return f"{name}_gt"

def error_name(name):
    return f"{name}_error"

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

def split_sets(ratio, num_series):
    test_size = max(int(ratio[2] * num_series), 1)
    val_size = max(int(ratio[1] * num_series), 1)
    train_size = num_series - test_size - val_size

    indices = list(range(num_series))
    random.shuffle(indices)
    test_idx = [indices.pop(0) for _ in range(test_size)]
    val_idx = [indices.pop(0) for _ in range(val_size)]
    train_idx = [indices.pop(0) for _ in range(train_size)]
    return train_idx, val_idx, test_idx

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

def save_dataset(df, data_config, path = "data/"):
    name = data_config["name"]
    if check_overwrite(name, path):
        savepath = f"{path}{name}"
        try:
            shutil.rmtree(savepath)
        except FileNotFoundError:
            pass
        print(f"saved dataset to: {savepath}")
        os.makedirs(savepath, exist_ok=True)
        df.to_csv(f"{savepath}/{name}.csv")

        with open(f'{savepath}/{name}.json', 'w') as fp:
            json.dump(data_config, fp, indent=6)

def load_dataset(name, path = "data/"):
    savepath = f"{path}{name}"
    df = pd.read_csv(f"{savepath}/{name}.csv", index_col=[0, 1])
    with open(f'{savepath}/{name}.json', 'r') as stream:
        config = json.load(stream)
    return df, config

def load_data(name, suff, path = "data/"):
    savepath = f"{path}{name}"
    df = pd.read_csv(f"{savepath}/{name}_{suff}.csv", index_col=[0, 1])
    return df

def load_result(name, variation, suff, path = "results/"):
    savepath = f"{path}{name}"
    df = pd.read_csv(f"{savepath}/{name}_{variation}_prediction_{suff}.csv", index_col=[0, 1])
    return df

def save_data(df, name, suff, path = "data/"):
    savepath = f"{path}{name}"
    df.to_csv(f"{savepath}/{name}_{suff}.csv")

def load_json(name, suff, path = "data/"):
    savepath = f"{path}{name}"
    with open(f'{savepath}/{suff}.json', 'r') as stream:
        config = json.load(stream)
    return config

def prepare_folder(name, path = "models/"):
    if check_overwrite(name, path):
        savepath = f"{path}{name}"
        try:
            shutil.rmtree(savepath)
        except FileNotFoundError:
            pass
        print(f"created {savepath}")
        os.makedirs(savepath, exist_ok=True)

def save_model(savepath, model_state, model_config):
    model_name = model_config["name"]
    torch.save(model_state, f"{savepath}/{model_name}_model.pt")

    with open(f'{savepath}/{model_name}_config.json', 'w') as fp:
        json.dump(model_config, fp, indent=6)

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

def gen_spike(when, height, length):
    """
    create spike impulse at certain (percent of total length [length]) points [when]
    and height [height]
    """

    out = []
    for w, h  in zip(when, height):
        data = np.zeros(length)
        for i in range(len(w)):
            pos = int(w[i]*length)
            data[pos] = h[i]
        out.append(list(data))

    return out

def gen_slope(start, end, length):
    slope = (end - start)/(length-1)

    res = []
    for i in range(length):
        res.append(start + slope * i)
    
    return res

def gen_saw(when, height, length):
    """
    create step impulse at certain (percent of total length [length]) points [when]
    and height [height]
    """

    out = []
    for w, h  in zip(when, height):
        data = np.zeros(length)
        for i in range(len(w)):
            start = int(h[i][0])
            end = int(h[i][1])
            w0 = w[i][0]
            w1 = w[i][1]
            pos1 = int(w0*length)
            pos2 = int(w1*length)
            
            period = pos2 - pos1
            
            tmp = gen_slope(start, end, period)

            data[pos1:pos2] = tmp
            data[pos2:] = end
        out.append(list(data))

    return out

# inputs: jump points, 
def gen_input(data_config):

    mem = {}

    samples = data_config["samples"]
    for inp, val in data_config["input_config"].items():
        mem[inp] = []
        for type, desc in val["types"].items():
            print(f"generating {type} with {desc}")

            # step function
            if type == "steps":
                data = gen_step(desc["when"], desc["height"], samples)
                if len(np.shape(data)) == 2:
                    mem[inp] += data
                else:
                    mem[inp].append(data)

            # saw function
            if type == "saw":
                data = gen_saw(desc["when"], desc["height"], samples)
                if len(np.shape(data)) == 2:
                    mem[inp] += data
                else:
                    mem[inp].append(data)

            # spikes function
            if type == "spikes":
                data = gen_spike(desc["when"], desc["height"], samples)
                if len(np.shape(data)) == 2:
                    mem[inp] += data
                else:
                    mem[inp].append(data)

            # custom functions just takes lists given in custom
            if type == "custom":
                mem[inp].append(desc)

    # get all combinations of input variations
    comb = list(itertools.product(*mem.values()))
    df = pd.DataFrame()
    for idx, sample in enumerate(comb):
        index = [(idx, i) for i in range(samples)]
        index = pd.MultiIndex.from_tuples(index, names=["series", "index"])
        out_df = pd.DataFrame(np.array(sample).T, index= index, columns=data_config["inputs"])
        df = pd.concat([df,out_df])

    return df

def gen_data(data_config, func):
    """
    generate ode results from inputs and other config options set in config dict, and returns it as Dataframe
    """

    # generate input dataframe and read constants, need to change to work with multiple constants
    input_df = gen_input(data_config)
    constants = tuple(data_config["constants"].values())

    # init counter and resulting Dataframe
    counter = 0
    result = pd.DataFrame()

    # loop over "series" index
    for _, data in input_df.groupby(level=0):
        # loop over different initial conditions given
        for init in data_config["init"]:

            print(f"generating {counter}th series from input")
            # transform "inputs" array to fit in ode generator
            inputs = data[data_config["inputs"]].T.values.tolist()
            inputs = np.array(inputs).T

            # calculate ode results for one current series
            x = Gen(func,(inputs,), constants, init,data_config["timestep"],data_config["samples"])
            x.generate()
            x.transform()

            # create new multiindex for dataframe
            index = [(counter, i) for i in range(data_config["samples"])]
            index = pd.MultiIndex.from_tuples(index, names=["series", "index"])

            # fill dataframe with new data and concat it with resulting dataframe to stack them above each other
            df = pd.DataFrame(np.array(x.X), index= index, columns=data_config["outputs"] + data_config["inputs"])
            result = pd.concat([result,df])

            counter += 1

    return result

def create_multiindex(pred, gt, data_config, model_config):

    result = pd.DataFrame()

    sequences = pred.shape[0]
    samples = pred.shape[1]

    for seq in range(sequences):
        # create new multiindex for dataframe
        index = [(seq, i) for i in range(samples)]
        index = pd.MultiIndex.from_tuples(index, names=["series", "index"])

        if model_config["norm"]:
            pred_names = [norm_name(pred_name(x)) for x in data_config["outputs"]]
            gt_names = [norm_name(gt_name(x)) for x in data_config["outputs"]]
        else:
            pred_names = [pred_name(x) for x in data_config["outputs"]]
            gt_names = [gt_name(x) for x in data_config["outputs"]]

        df = pd.DataFrame(np.hstack((pred[seq],gt[seq])), index= index, columns=pred_names + gt_names)
        result = pd.concat([result,df])

    if model_config["norm"]:
        result = inv_scale_results(result, data_config)
    return result

def scale(df, data_config, input_scaler, output_scaler):
    df[[f"{x}_norm" for x in input_scaler.get_feature_names_out()]] = input_scaler.transform(df[data_config["inputs"]])
    df[[f"{x}_norm" for x in output_scaler.get_feature_names_out()]] = output_scaler.transform(df[data_config["outputs"]])
    return df

def inv_scale_results(results, data_config):

    result_scaler = joblib.load(f"data/{data_config['name']}/{data_config['name']}_output_scaler.pkl")
    results[[f"{pred_name(x)}" for x in result_scaler.get_feature_names_out()]] = result_scaler.inverse_transform(results[[norm_name(pred_name(x)) for x in result_scaler.get_feature_names_out()]])
    results[[f"{gt_name(x)}" for x in result_scaler.get_feature_names_out()]] = result_scaler.inverse_transform(results[[norm_name(gt_name(x)) for x in result_scaler.get_feature_names_out()]])

    return results

def sort_paths(paths):
    """
    sort paths by hyperparameter descriptions in filename
    all need to be numbers, but arch description can be string, made exception below
    this is really ugly, but works for now
    """
    configs = {}
    for path in paths:
        with open(path, "r") as f:
            configs[path] = json.load(f)

    # replacing arch description with number to sort it
    sizes = {"OneLayers":1, "TwoLayers":2, "ThreeLayers":3, "FourLayers":4, "FiveLayers":5}
    for config in configs.values():
        config["arch"] = sizes[config["arch"]]
    
    # create df with all hyper values and path
    temp = pd.DataFrame()
    for key, value in configs.items():
        hyper_desc = value["hyper_desc"]
        cols = ["path"]
        cols += hyper_desc
        values = [[key]]
        values[0] += [value[x] for x in hyper_desc]
        df = pd.DataFrame(values, columns=cols)
        temp = pd.concat([temp,df])

    hyper_desc = configs[list(configs.keys())[0]]["hyper_desc"]
    temp.sort_values(hyper_desc, inplace=True)
    sorted_paths = list(temp["path"].values)
    return sorted_paths

def create_summary(experiment_name, models_path="models/"):

    """
    create summary of all models in experiment
    plot them, and save table with results
    """

    # make dpi of matplotlib 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # get paths of all model_configs
    paths = glob.glob(f"{models_path}{experiment_name}/*_config.json")

    paths = sort_paths(paths)

    # create summary dataframe
    summary = pd.DataFrame()
    legend_params = []
    for path in paths:
        # read all information for model
        # load model config
        with open(path, 'r') as stream:
            model_config = json.load(stream)
        with open(f"{models_path}{experiment_name}/{model_config['name']}_best.json", 'r') as stream:
            best_info = json.load(stream)
        
        # add info as row to summary dataframe
        columns = ["name"] + model_config["hyper_desc"]
        values = [[model_config[x] for x in columns]]
        values[0].append(best_info["best_val_loss"])
        values[0].append(best_info["best_epoch"])
        columns += ["val_loss", "best_epoch"]
        df  = pd.DataFrame(values, columns=columns)
        summary = pd.concat([summary,df])
        summary.reset_index(drop=True, inplace=True)
        
        # load loss csv file
        loss_hist = pd.read_csv(f"{models_path}{experiment_name}/{model_config['name']}_loss.csv", index_col=0)

        # save list with descriptions of models to later use as legend
        param_desc = "; ".join([f"{desc}: {model_config[desc]}" for desc in model_config["hyper_desc"]])
        legend_params.append(param_desc)

        plt.figure(1)
        loss_hist["train_loss"].plot(linewidth=0.5)

        plt.figure(2)
        loss_hist["val_loss"].plot(linewidth=0.5)

        plt.figure(3)
        loss_hist.plot(linewidth=0.5)
        ylabel = "Loss"
        xlabel = "Epoch"
        plt.yscale("log")
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(f"{models_path}{experiment_name}/{model_config['name']}_loss.pdf")
        plt.savefig(f"{models_path}{experiment_name}/{model_config['name']}_loss.png")
        plt.close()
        plt.clf()
    
    # create folder for summary
    os.makedirs(f"{models_path}{experiment_name}/summary", exist_ok=True)

    # save summary df as csv
    summary.to_csv(f"{models_path}{experiment_name}/summary/summary.csv")

    # save loss summary plots
    fontsize = 5
    ylabel = "Loss"
    xlabel = "Epoch"
    plt.figure(1)
    plt.legend(legend_params, prop={'size': fontsize})
    plt.yscale("log")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(f"{models_path}{experiment_name}/summary/{experiment_name}_train_loss_comparison.pdf")
    plt.savefig(f"{models_path}{experiment_name}/summary/{experiment_name}_train_loss_comparison.png")

    plt.figure(2)
    plt.legend(legend_params, prop={'size': fontsize})
    plt.yscale("log")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(f"{models_path}{experiment_name}/summary/{experiment_name}_val_loss_comparison.pdf")
    plt.savefig(f"{models_path}{experiment_name}/summary/{experiment_name}_val_loss_comparison.png")

    # get best model name and min loss from summary df
    min_loss = summary["val_loss"].min()
    best_name = summary.loc[summary["val_loss"].idxmin()]["name"]
    best_epoch = summary.loc[summary["val_loss"].idxmin()]["best_epoch"]

    best_model = {
            "best_model_name": best_name,
            "min_loss": min_loss,
            "epoch": int(best_epoch),
            }

    with open(f"{models_path}{experiment_name}/summary/best_model.json", 'w') as fp:
        json.dump(best_model, fp, indent=6)

# check if list has one element, if so return element, else return list
def c_one(l):
    if len(l)==1:
        return l[0]
    else:
        return l

def get_entries(df, amount):
    indices = df.index.unique(level="series")
    indices = indices[:amount]
    return df.loc[indices]

if __name__ == "__main__":
    # dataset to load
    dataset_name = "pend_test"

    # define experiment identifiers
    descripor = "test"
    version = "2"

    # create full name for folder containing experiment
    experiment_name = f"{dataset_name}_{descripor}_{version}"

    create_summary(experiment_name)
