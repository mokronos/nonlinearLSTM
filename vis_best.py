import matplotlib.pyplot as plt
import pandas as pd
from helper import load_dataset


name = "pendulum_3init0force"
df, config = load_dataset(name)

for i in range(len(set(df.index.get_level_values(0)))):
    df.xs(i)[config["outputs"]].plot()
    plt.legend(config["output_labels"])
    plt.xlabel("time in 0.01s")
    plt.ylabel("rad/ rad/s")

plt.show()
