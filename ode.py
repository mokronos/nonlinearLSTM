import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt

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
timestep = 0.01
t = np.arange(0, samples*timestep, timestep)

# y = solve_ivp(model, [0,samples*timestep], y0, dense_output=True, method = "LSODA")
# y = y.sol(t)
# y1 = y

# y[0] = angle' y[1] = angle
# plt.figure(1)
# plt.plot(t,y[1,:])

# plt.show()

class Gen:
    """
    Generator for input and output data from a given differential equation
    """

    def __init__(self, func, parameters, y0, timestep, samples):

        self.func = func
        self.parameters = parameters
        self.y0 = y0
        self.timestep = timestep
        self.samples = samples
        self.rng = np.random.default_rng(seed=42)
    
    def step(self): 
        
        t = np.arange(0, self.timestep*2, self.timestep)
        result = solve_ivp(self.func, [0, self.timestep*2], self.y0, dense_output=True, method = "LSODA")
        result = result.sol(t)
        result = result[:,-1]
        self.y0 = result
        return result

    def generate(self):
        x = []
        for _ in range(self.samples):
            x.append(self.step())

        x = np.array(x)
        x = self.add_noise(x)
        X,y = self.gen_samples(x)
        
        return X, y

    def add_noise(self, data):

        noise = self.rng.normal(0, 0.01, data.shape)
        return data + noise

    def gen_samples(self, data):

        X = []
        y = []

        self.length = 5

        for i in range(self.samples - self.length - 1):

            timeseries = []
            for j in range(i,i+self.length):
                timeseries.append(data[j, 1])
            X.append(timeseries)
            y.append([data[i+self.length + 1, 1]])
        
        return np.array(X), np.array(y)


x = Gen(model,2,y0,timestep,samples)

X,y = x.generate()

print(X, y)
print(X.shape, y.shape)
# t = np.arange(0, samples*timestep, timestep)
# plt.figure(2)
# plt.plot(t,y[:,1])
# plt.show()
