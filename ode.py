import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

class Gen:
    """
    Generator for input and output data from a given differential equation
    """

    def __init__(self, func, parameters, y0, dt, samples):

        self.func = func
        self.parameters = parameters
        self.y0 = y0
        self.dt = dt
        self.samples = samples
        self.rng = np.random.default_rng(seed=42)
    
    def step(self): 
        
        t = np.arange(0, self.dt*2, self.dt)
        result = solve_ivp(self.func, [0, self.dt*2], self.y0, dense_output=True, method = "LSODA")
        result = result.sol(t)
        result = result[:,-1]
        self.y0 = result
        return result

    def generate(self):
        data = []
        for _ in range(self.samples):
            data.append(self.step())

        self.raw = np.array(data)
        self.add_noise

    def add_noise(self):

        noise = self.rng.normal(0, 0.01, self.raw.shape)
        self.meas = self.raw + noise


class timeseries(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    
    # pendulum diff. equation
    # y'' + (g/l) * sin(y) = 0
    # split into two first order diff. equations 
    # y[0] = y
    # y[1] = y'
    # --> y[0]' = y[1]
    # --> y[1]' = - (g/l) * sin(y[0])
    
    def model(t,y):
        l = 1
        g = 9.8
    
        res = [y[1], -(g/l) * np.sin(y[0])]
        return res
    
    # [angle'_init, angle_init]
    y0 = [0, 0.1]
    samples = 500
    dt = 0.01
    
    x = Gen(model,None,y0,dt,samples)
    x.generate()
    t = np.arange(0, len(x.raw)*dt, dt)
    
    plt.plot(t,x.raw)
    plt.show()
