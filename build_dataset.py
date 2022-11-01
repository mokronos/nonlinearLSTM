from helper import load_dataset, split_sets

# normalize train set, use same normalization on val and test, only input vector
def min_max_scale(df, col, min, max):

    # special case, produces nan otherwise (divide by 0)
    if min == max:
        df[f"{col}_norm"] = df[col]*0

    min_max_scaler = lambda x: (x-min)/(max-min)

    df[f"{col}_norm"] = df[col].apply(min_max_scaler)

    return df

def inverse_min_max_scale(df, col, min, max):

    # x: (x-min)/(max-min)
    # x: x * (max-min) + min
    # x: x*max - x*min + min

    # special case, produces nan otherwise (divide by 0)
    min_max_scaler = lambda x: x*max - x*min + min

    df[f"{col}_invnorm"] = df[col].apply(min_max_scaler)

    return df


def scale(df, stats):

    for label in df.columns:

        min = stats[label]["min"]
        max = stats[label]["max"]

        df = min_max_scale(df, label, min, max)
    return df

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
    train_stats = train_df.describe()

    # save normalized values in addition to normal values
    # and save normalization values to reverse later
    train_df_scaled = scale(train_df, train_stats)
    val_df_scaled = scale(val_df, train_stats)
    test_df_scaled = scale(test_df, train_stats)

    # save all 3
    train_df_scaled.to_csv(f"{savepath}/{name}_train.csv")
    val_df_scaled.to_csv(f"{savepath}/{name}_val.csv")
    test_df_scaled.to_csv(f"{savepath}/{name}_test.csv")

    # save train_stats dataframe to later invert the normalization easily (and get some stats)
    train_stats.to_csv(f"{savepath}/{name}_train_stats.csv")

if __name__ == "__main__":

    name = "drag_step"
    build_dataset(name)
