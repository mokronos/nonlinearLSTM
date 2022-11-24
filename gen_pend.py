import numpy as np
import matplotlib.pyplot as plt
import random
from helper import load_dataset, save_dataset, gen_data
from build_dataset import build_dataset
from random import randint, choice, randrange

# theta'' + q*theta' + (g/l) * sin(theta) = F_p*cos(freq*t)
# a_ang = theta''
# v_ang = theta'
# p_ang = theta
#
# 1. theta' = v_ang (angular velocity) --> angular position
# 2. theta'' = F_p * cos(freq*t) - q * v_ang - (g/l) * sin(theta) (angular acceleration) --> angular velocity

random.seed(3)

def pend(y,t,u, g, l, q, freq):
    # l = 1
    # res = [y[1], -(g/l) * np.sin(y[0]) + u]

    # better names for input(pushing force) and starting condition(velocity)
    f_p = u
    p_0, v_0 = y

    theta = v_0
    # should have been f_p/(m*l) (but m = l = 1, so in my experiments it doesnt matter)
    theta_v = f_p * np.cos(freq*t) - q * v_0 - (g/l) * np.sin(p_0)
    return [theta, theta_v]

# fix random seed
random.seed(3)

# pend_simple
# init = [[0, x] for x in np.arange(-2,2.0,0.1)]
# spikes_height = [[0, x*0.7,x *0.5,x*0.6] for x in range(100,310,10)]
# spikes_when = [[0,0.1,0.2,0.4]]


# pend_simple_var
amount = 5
init = [[0, x] for x in np.arange(-2,2.1,0.5)]
spikes_height = []
for _ in range(amount):
    tmp = [randrange(-250,251,50),randrange(-250,251,50),randrange(-250,251,50)]
    spikes_height.append(tmp)
spikes_when = [[0.2,0.5,0.7]]*amount

# pend_complex
# amount = 20

# init = [[0, x] for x in np.arange(-2,2.1,1)]
# spikes_height = []
# for _ in range(amount):
#     tmp = [randrange(-200,201,50)]
#     for i in range(2):
#         tmp.append(tmp[i] + randrange(-70,71,10))
#     spikes_height.append(tmp)

# spikes_when = []
# for _ in range(amount):
#     tmp = [choice([0.1,0.2,0.3]),choice([0.4,0.5,0.6]),choice([0.7,0.8,0.9])]
#     spikes_when.append(tmp)

# step_height = []
# for _ in range(amount):
#     tmp = [randrange(-5,6,1)]
#     for i in range(2):
#         tmp.append(tmp[i] + randrange(-1,2,1))
#     step_height.append(tmp)

# step_when = []
# for _ in range(amount):
#     tmp = [choice([0.1,0.2,0.3]),choice([0.4,0.5,0.6]),choice([0.7,0.8,0.9])]
#     step_when.append(tmp)

config = {
        "name":"pend_simple_var",
        "function": "pend",
        "samples": 1000,
        "train_val_test_ratio" : [0.6, 0.2, 0.2],
        "time": "t",
        "outputs": ["theta","d_theta"],
        "output_labels":[r"\theta",r"\dot{\theta}"],
        "output_units":["rad","rad/s"],
        "error_units":["rad^2","rad^2/s^2"],
        "input_units":["N"],
        "input_labels":[r"F_{p}"],
        "init": init,
        "timestep": 0.01,
        "constants": {
                "g": 9.81,
                "l": 1,
                "q": 1,
                "freq": 2
                },
        "input_config": {
                "force_inp":{
                    "types": {
                        # "steps": {
                        #     "when": step_when,
                        #     "height": step_height
                        #     },
                        "spikes": {
                            "when": spikes_when,
                            "height": spikes_height
                            },
                        }}}
        } 

config["inputs"] = list(config["input_config"].keys())


data = gen_data(config, eval(config["function"]))
save_dataset(data, config)

build_dataset(config["name"])
df, config = load_dataset(config["name"])

df.loc[0].plot()
plt.show()
