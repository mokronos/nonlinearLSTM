import numpy as np
import matplotlib.pyplot as plt
from helper import load_dataset, save_dataset, gen_data
from build_dataset import build_dataset


data_path = "data/"


# speed: v = v_0 + a * t
# position: z = z_0 + 1/2 * a * t**2 + v_0 * t 
# equation for free fall or pushing something
# F_P = m*a | force of push (gravity)
# F_D = 1/2 * p * v ** 2 * C_D * A | Drag Force
# with b = 1/2 * p * C_D * A
# F_D = b * v ** 2
#
# v = dz/dt
# a = d**2 z/d**2 t
# F = F_P - F_D
# diff. equation:
# m * d2z/d2t = F_P - b * (dz/dt)**2
# use v = dz/dt
# 1. dz/dt = v (speed)
# 2. dv/dt = (F_P - b * v**2)/m (acceleration)

def drag(y,t,u,m,c_d,rho,area):
    
    # calculate constant drag coefficient
    b = 1/2 * rho * c_d * area

    # better names for input(pushing force) and starting condition(velocity)
    f_p = u
    v_0 = y
    # if position is needed too:
    # z_0, v_0 = y

    # if position is needed too:
    # dz = v_0
    dv = (f_p - b * np.square(v_0))/m

    # if position is needed too:
    # return [dz,dv]
    return dv

x = [[0]]*100
y = [[0, x*1.5**1, x*1.5**2,x*1.5**3] for x in range(10,210,10)]

config = {
        "name":"drag_mult_step",
        "function": "drag",
        "samples": 3000,
        "train_val_test_ratio" : [0.6, 0.2, 0.2],
        "time": "t",
        "outputs": ["velocity"],
        "output_labels":[r"$v$"],
        "output_units":[r"m/s"],
        "input_units":[r"N"],
        "input_labels":[r"$F_{p}$"],
        "init": [[0]],
        "timestep": 0.01,
        "constants": {
                "m": 5.436,
                "c_d": 0.5,
                "rho": 1.2,
                "area": 0.216,
                },
        "input_config": {
                "inp_acc":{
                    "types": {
                        "steps": {
                            "when":
                            [[0,0.1,0.3,0.6]]*20,
                            "height":
                            y
                            },
                        }},
        }
        }
config["inputs"] = list(config["input_config"].keys())

data = gen_data(config, eval(config["function"]))
save_dataset(data, config)

build_dataset(config["name"])
df, config = load_dataset(config["name"])

df.loc[0].plot()
plt.show()
