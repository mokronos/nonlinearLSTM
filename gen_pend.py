import numpy as np
import matplotlib.pyplot as plt
from helper import load_dataset, save_dataset, gen_data
from build_dataset import build_dataset

def pend(y,t,u,g):
    l = 1

    res = [y[1], -(g/l) * np.sin(y[0]) + u]
    return res

x = [[0, x] for x in np.arange(-2,2,0.02)]
config = {
        "name":"pend_mult",
        "function": "pend",
        "samples": 200,
        "train_val_test_ratio" : [0.6, 0.2, 0.2],
        "time": "t",
        "outputs": ["d_theta","theta"],
        "output_labels":[r"$\dot{\theta}$",r"$\theta$"],
        "output_units":[r"m/s"],
        "input_units":[r"N"],
        "input_labels":[r"$F_{p}$"],
        "init": x,
        "timestep": 0.01,
        "constants": {
                "g": 9.81
                },
        "input_config": {
                "force1":{
                    "types": {
                        "steps": {
                            "when": [[0]],
                            "height": [[0]]
                        }}}}
        } 

config["inputs"] = list(config["input_config"].keys())


data = gen_data(config, eval(config["function"]))
save_dataset(data, config)

build_dataset(config["name"])
df, config = load_dataset(config["name"])

df.loc[0].plot()
plt.show()
