import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from helper import create_dataset, create_multiindex, load_data, load_json, norm_name, prepare_folder, save_data, get_mse
from models import *
from train import test_loop

def predict_best_model(data_name, descriptor, version, variation = "base"):
    # set random seed
    torch.manual_seed(3)

    #################################################
    # load dataset for test data and variable information
    name = data_name
    data_config = load_json(name, name)
    df_train = load_data(name, "train")
    df_val = load_data(name, "val")
    df_test = load_data(name, "test")

    # define experiment identifiers
    descriptor = descriptor
    version = version
    name = data_config["name"]
    # create full name for folder containing experiment
    experiment_name = f"{name}_{descriptor}_{version}"

    # prepare folder for saving results
    result_dir = "results/"
    variation = variation
    prepare_folder(experiment_name, result_dir)

    # load json to figure out what model is the best one
    model_dir = "models/"
    loadpath = f"{model_dir}{experiment_name}"
    with open(f"{loadpath}/summary/best_model.json", 'r') as stream:
        best_model = json.load(stream)

    # load best model config
    with open(f"{loadpath}/{best_model['best_model_name']}_config.json", 'r') as stream:
        model_config = json.load(stream)

    samples = data_config["samples"]
    batch_size = model_config["bs"]

    # define which column of data to train on depending on if normalization is used
    if model_config["norm"]:
        input_names = [norm_name(x) for x in data_config["inputs"]]
        output_names = [norm_name(x) for x in data_config["outputs"]]
    else:
        input_names = data_config["inputs"]
        output_names = data_config["outputs"]


    #################################################
    # create NN and train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")

    # load model
    model_ext = ".pt"
    model_path = f"{model_dir}{experiment_name}/{best_model['best_model_name']}_model{model_ext}"

    # create model
    # number of features
    input_size = len(data_config["inputs"]) + len(data_config["outputs"])

    # define hidden nodes, pack in list so network can be variable depth (rest gets ignored)
    # if nodes not None, make all layers have same number of nodes
    nodes = model_config["nodes"]
    if nodes:
        h1 = h2 = h3 = h4 = h5 = nodes
    else:
        h1 = model_config["h1"]
        h2 = model_config["h2"]
        h3 = model_config["h3"]
        h4 = model_config["h4"]
        h5 = model_config["h5"]
    hidden_nodes = [h1, h2, h3, h4, h5]

    # number of outputs
    output_size = len(data_config["outputs"])

    model = eval(model_config["arch"])(input_size, output_size, *hidden_nodes).to(device)
    print(model)

    loss_fn = torch.nn.MSELoss()

    model.load_state_dict(torch.load(model_path))
    model.eval()
    data = {"train":df_train, "val": df_val,"test": df_test}

    loss_mem = []
    mse_mem = []

    for desc, data in data.items():

        ds = create_dataset(data, input_names, output_names, 1, samples - 1)

        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        predictions = []
        ground_truth = []
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            # measure time to predict one sample
            with torch.no_grad():
                start_time = time.time()
                pred = model(X).detach().cpu().numpy()
                end_time = time.time()
                print(f"Time to predict: {end_time - start_time} seconds")

            gt = y.detach().cpu().numpy()
            predictions.append(pred)
            ground_truth.append(gt)

        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        predictions = predictions.reshape((-1,)+predictions.shape[-2:])
        ground_truth = ground_truth.reshape((-1,)+predictions.shape[-2:])


        results = create_multiindex(predictions, ground_truth, data_config, model_config)

        save_data(results, experiment_name, f"{variation}_prediction_{desc}", path=result_dir)

        loss, _ = test_loop(dataloader, model, loss_fn, device)

        mse_mem.extend(get_mse(results,data_config, desc))
        str = f"loss on {desc}set: {loss}"
        loss_mem.append(str)
    
    # write loss values to file
    with open(f"{result_dir}{experiment_name}/{variation}_losses.txt", 'w') as stream:
        stream.write("\n".join(loss_mem))
        stream.write("\n")
        stream.write("\n".join(mse_mem))

if __name__ == "__main__":

    # dataset_name = "drag_simple_steps"
    # dataset_name = "drag_complex"
    # dataset_name = "drag_complex_var"
    # dataset_name = "pend_simple"
    # dataset_name = "pend_simple_var"
    dataset_name = "pend_complex"
    descriptor = "alpha"
    version = 4
    variation = "base"
    predict_best_model(dataset_name, descriptor, version, variation)
