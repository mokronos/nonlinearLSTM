import pandas as pd
import numpy as np

from data import Gen



def func(y,t,u,g):
    l = 1

    # res = [y[1], -(g/l) * np.sin(y[0]) + u[0] + u[1]]
    res = [y[1], -(g/l) * np.sin(y[0]) + u[0]]
    return res

# [angle'_init, angle_init]
y0 = [0, 0.1]
samples = 3000
dt = 0.01

# if parameters should vary in time, give them as lists like this for [[par1(t=0),par2(t=0)], [par1(t=1),par2(t=1)], ... ]; just append them as list, then transpose
# if single constant, just give them as one value
u = []
u.append([0]*samples)
# u.append([np.sin(x) for x in np.arange(0, samples*dt,dt)])
u = np.array(u)
u = u.T
g = 9.81

# give input, then parameters (both as tuples); inputs = things the RNN/model gets as input as well, parameters = things the model is supposed to learn (potential changes in the system)
x = Gen(func,(u,),(g,),y0,dt,samples)
x.generate()
x.transform()

print(x.X.shape)
f = pd.DataFrame(x.X)
print(f.head())
