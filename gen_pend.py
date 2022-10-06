import numpy as np
from helper import  gen_data, save_dataset

def pend(y,t,u,g):
    l = 1

    res = [y[1], -(g/l) * np.sin(y[0]) + u]
    return res

config = {
        "name":"pendulum_3init0force",
        "function": "pend",
        "samples": 1000,
        "time": "t",
        "outputs": ["d_theta","theta"],
        "output_labels":[r"$\dot{\theta}$",r"$\theta$"],
        "init": [[0, 0.1],[0, 0.2],[0,0.25]],
        "timestep": 0.01,
        "constants": {
                "g": 9.81
                },
        "inputs": {
                "force1":{
                    "types": {
                        "steps": {
                            "when": [[0]],
                            "height": [[0]]
                        }}}}
        } 



data = gen_data(config, pend)
save_dataset(data, config)
