import matplotlib.pyplot as plt
import pandas as pd
from helper import load_dataset

name = "drag_step"
df, config = load_dataset(name)

# for i in range(len(set(df.index.get_level_values(0)))):
#     df.xs(i)[config["outputs"]].plot()
#     plt.legend(config["output_labels"])
#     plt.xlabel("time in 0.01s")
#     plt.ylabel(fr'{config["outputs"][0]} in {config["output_units"][0]}')

info = [f"{key}: {value}" for key, value in config["constants"].items()]
info = "; ".join(info)

for i in range(len(set(df.index.get_level_values(0)))):
    fig, ax1 = plt.subplots()
    fig.suptitle(info)
    ax2 = ax1.twinx()
    color1 = "g"
    color2 = "r"
    df.xs(i)[config["outputs"]].plot(ax=ax1, color = color1)
    df.xs(i)[list(config["inputs"].keys())].plot(ax=ax2, color = color2)
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    ax1.legend(config["output_labels"], loc="upper left")
    ax2.legend(config["input_labels"], loc="lower right")
    ax1.set_xlabel(r"time in $0.01 s$")
    ax1.set_ylabel(fr'{config["output_labels"][0]} in {config["output_units"][0]}', color = color1)
    ax2.set_ylabel(fr'{config["input_labels"][0]} in {config["input_units"][0]}', color = color2)

# amount = 20
# df = df.loc[0:amount]
# plt.subplots(20)
# for i in range(len(set(df.index.get_level_values(0)))):
#     fig, ax1 = plt.subplots()
#     fig.suptitle(info)
#     ax2 = ax1.twinx()
#     color1 = "g"
#     color2 = "r"
#     df.xs(i)[config["outputs"]].plot(ax=ax1, color = color1)
#     df.xs(i)[list(config["inputs"].keys())].plot(ax=ax2, color = color2)
#     ax1.legend(config["output_labels"], loc="upper left")
#     ax2.legend(config["input_labels"], loc="lower right")
#     ax1.set_xlabel(r"time in $0.01 s$")
#     ax1.set_ylabel(fr'{config["output_labels"][0]} in {config["output_units"][0]}', color = color1)
#     ax2.set_ylabel(fr'{config["input_labels"][0]} in {config["input_units"][0]}', color = color2)
plt.show()
