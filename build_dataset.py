from helper import load_data, load_dataset, split_sets, gt_name, pred_name
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
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
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

def scale(df, data_config, input_scaler, output_scaler):
    df[[f"{x}_norm" for x in input_scaler.get_feature_names_out()]] = input_scaler.transform(df[data_config["inputs"]])
    df[[f"{x}_norm" for x in output_scaler.get_feature_names_out()]] = output_scaler.transform(df[data_config["outputs"]])
    return df

def inv_scale_results(results, data_config):

    result_scaler = joblib.load(f"data/{data_config['name']}/{data_config['name']}_outputs_scaler.pkl")

    results[[f"{pred_name(x)}_invnorm" for x in result_scaler.get_feature_names_out()]] = result_scaler.inverse_transform(results[[pred_name(x) for x in result_scaler.get_feature_names_out()]])
    results[[f"{gt_name(x)}_invnorm" for x in result_scaler.get_feature_names_out()]] = result_scaler.inverse_transform(results[[gt_name(x) for x in result_scaler.get_feature_names_out()]])

    return results

if __name__ == "__main__":

    name = "drag_mult_step"
    df, data_config = load_dataset(name)
    df = df.loc[:1]
    # build_dataset(name)
    scaled_df = scale(df, data_config)
    # experiment name
    descriptor = "wholeseries_normed"
    version = "1"
    # create full name for folder containing experiment
    experiment_name = f"{data_config['name']}_{descriptor}_{version}"
    suffixes = ["train", "val", "test"] 
    suffix = suffixes[0]
    results = load_data(experiment_name, f"prediction_{suffix}", path="results/")
    results = inv_scale_results(results, data_config)
    
