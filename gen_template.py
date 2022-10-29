import numpy as np
import matplotlib.pyplot as plt
from helper import load_dataset, save_dataset, gen_data


data_path = "data/"


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

df, config = load_dataset(config["name"])

df.plot()
plt.show()
