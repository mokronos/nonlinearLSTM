from helper import load_dataset, load_result, split_sets, scale
import json
from sklearn.preprocessing import MinMaxScaler
import joblib


def build_dataset(name, path = "data/"):

    # generate savepath for later
    savepath = f"{path}{name}"

    # load the from odes generated data
    df, data_config = load_dataset(name)
    
    # split into train, val, test set
    ratio = data_config["train_val_test_ratio"]
    num_series = len(df.groupby(level=0))

    # get randomized indices split by ratio
    train_idx, val_idx, test_idx = split_sets(ratio, num_series)

    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]

    # fit scaler for input and output on train set to later retrieve only output scaler to inverse transform
    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    input_scaler.fit(train_df[data_config["inputs"]])
    output_scaler.fit(train_df[data_config["outputs"]])
    joblib.dump(input_scaler, f"data/{data_config['name']}/{data_config['name']}_input_scaler.pkl")
    joblib.dump(output_scaler, f"data/{data_config['name']}/{data_config['name']}_output_scaler.pkl")

    # scale train, val, test set
    # save normalized values in addition to normal values
    train_df_scaled = scale(train_df, data_config, input_scaler, output_scaler)
    val_df_scaled = scale(val_df, data_config, input_scaler, output_scaler)
    test_df_scaled = scale(test_df, data_config, input_scaler, output_scaler)

    # save all 3
    train_df_scaled.to_csv(f"{savepath}/{name}_train.csv")
    val_df_scaled.to_csv(f"{savepath}/{name}_val.csv")
    test_df_scaled.to_csv(f"{savepath}/{name}_test.csv")

    # get length of datasets
    ds_len = len(df.index.unique(level="series"))
    train_len = len(train_df.index.unique(level="series"))
    val_len = len(val_df.index.unique(level="series"))
    test_len = len(test_df.index.unique(level="series"))

    # update config with dataset lengths
    data_config["ds_len"] = ds_len
    data_config["train_len"] = train_len
    data_config["val_len"] = val_len
    data_config["test_len"] = test_len

    # save updated config
    with open(f"{savepath}/{data_config['name']}.json", "w") as fp:
        json.dump(data_config, fp, indent=6)

if __name__ == "__main__":

    name = "drag_mult_step"
    df, data_config = load_dataset(name)
    df = df.loc[:1]
    # build_dataset(name)
    # scaled_df = scale(df, data_config)
    # experiment name
    descriptor = "test"
    version = "2"
    # create full name for folder containing experiment
    experiment_name = f"{data_config['name']}_{descriptor}_{version}"
    suffixes = ["train", "val", "test"] 
    suffix = suffixes[0]
    results = load_result(experiment_name, "test")
    print(results)
    results = inv_scale_results(results, data_config)
    print(results)
    
