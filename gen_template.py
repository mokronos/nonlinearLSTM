import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from helper import create_dataset, load_dataset, save_dataset, gen_data


data_path = "data/"
ext = ".pkl"


def pend(y,t,u,g):
    l = 1

    res = [y[1], -(g/l) * np.sin(y[0]) + u[0] + u[1]]
    return res

config = {
        "name":"pendulum_simple",
        "function": "pend",
        "samples": 1000,
        "time": "t",
        "outputs": ["d_angle","angle"],
        "init": [[0, 0.1],[0,0.2]],
        "timestep": 0.01,
        "constants": {
                "g": 9.81
                },
        "inputs": {
                "force0":{
                    "types": {
                        "steps": {
                            "when":
                            [[0],
                             [0]],
                            "height":
                            [[0],
                             [1]]
                            },
                        "custom": [[1,2,3,4]]
                        }},
                "force1":{
                    "types": {
                        "steps": {
                            "when": [[0]],
                            "height": [[0]]
                        }}}}
        } 



data = gen_data(config, pend)
save_dataset(data, config)


df, config = load_dataset("pend_test")


torch.set_printoptions(precision=6)
df.plot()
plt.show()

data = create_dataset(df, config["inputs"].keys(), config["outputs"])
batch_size = 1
dl = DataLoader(data, batch_size=batch_size, shuffle=False)

for batch, (x,y) in enumerate(dl):
    print(f"batch:{batch}")
    print(f"x:{x}")
    print(f"y:{y}")
