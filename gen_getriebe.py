import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import Gen
from getriebe import HD_friction_compliance

data_path = "data/"
ext = ".pkl"
name = "getriebe"

# getriebe
# import ode model from getriebe.py file
func = HD_friction_compliance

# theta_wg, theta_fs, d_theta_wg, d_theta_fs
y0 = [0,0,0,0]
samples = 500
dt = 0.01

# if parameters should vary in time, give them as lists like this for [[par1(t=0),par2(t=0)], [par1(t=1),par2(t=1)], ... ]; just append them as list, then transpose
# if single constant, just give them as one value
i_m = []
# needs to be be 0.28 <= u <=0.6
i_m.append([0.28]*samples)
# u.append([np.sin(x) for x in np.arange(0, samples*dt,dt)])
i_m = np.array(u)
i_m = i_m.T

# give input, then parameters (both as tuples); inputs = things the RNN/model gets as input as well, parameters = things the model is supposed to learn (potential changes in the system)
x = Gen(func,(i_m,), (),y0,dt,samples)
x.generate()
x.transform()


f = pd.DataFrame(x.X, columns=["torque_in", "torque_out", "torque_in_d", "torque_out_d", "impulse"])
f.plot()
plt.show()

f.to_pickle(f"{data_path}{name}{ext}")
