import numpy as np
import matplotlib.pyplot as plt
from helper import gen_saw, load_dataset, save_dataset, gen_data
from build_dataset import build_dataset
import random
from random import randint, choice, randrange
from scipy.signal import sawtooth


data_path = "data/"
random.seed(3)


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
# 1. dz/dt = v (speed) --> position
# 2. dv/dt = (F_P - b * v**2)/m (acceleration) --> speed

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

# drag_simple_steps 
# init = [[0]]
# height = [[0, x*1.5**1, x*1.5**2,x*1.5**3] for x in range(10,210,10)]
# when = [[0,0.1,0.3,0.6]]*20

# drag_complex
init = [[x] for x in range(0, 10, 2)]

amount = 20
step_height = []
for _ in range(amount):
    tmp = [randrange(100,201,50)]
    for i in range(2):
        tmp.append(tmp[i] + randrange(-30,71,10))
    step_height.append(tmp)

step_when = []
for _ in range(amount):
    tmp = [choice([0.1,0.2,0.3]),choice([0.4,0.5,0.6]),choice([0.7,0.8,0.9])]
    step_when.append(tmp)

saw_height = []
for _ in range(amount):
    tmp = [(0,randrange(100,201,50))]
    for i in range(1):
        x = tmp[i][1]
        y = tmp[i][1] + randrange(-30,71,10)
        value = (x,y)

        tmp.append(value)
    saw_height.append(tmp)

saw_when = []
for _ in range(amount):
    tmp = [(choice([0.1,0.2]),choice([0.3,0.4])),(choice([0.6,0.7]),choice([0.8,0.9]))]
    saw_when.append(tmp)


config = {
        "name":"drag_complex",
        "function": "drag",
        "samples": 3000,
        "train_val_test_ratio" : [0.6, 0.2, 0.2],
        "time": "t",
        "outputs": ["velocity"],
        "output_labels":["v"],
        "output_units":["m/s"],
        "error_units":["m^2/s^2"],
        "input_units":["N"],
        "input_labels":[r"F_{p}"],
        "init": init,
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
                            step_when,
                            "height":
                            step_height
                            },
                        "saw": {
                            "when":
                            saw_when,
                            "height":
                            saw_height
                            },
                        }},
        }
        }
config["inputs"] = list(config["input_config"].keys())
# df["inp_acc"].loc[0].plot()

data = gen_data(config, eval(config["function"]))
save_dataset(data, config)

build_dataset(config["name"])
df, config = load_dataset(config["name"])

df["inp_acc"].plot()
plt.show()
