import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Gen:
    """
    Generator for input and output data from a given differential equation
    Outputs 2D array with states and inputs at every time step (state1, state2, input1, input2)
    """

    def __init__(self, func, inputs, parameters, y0, dt, samples):

        self.func = func
        self.inputs = inputs
        self.parameters = parameters
        self.y0 = y0
        self.dt = dt
        self.samples = samples
        self.rng = np.random.default_rng(seed=42)
    
    def generate(self):
        data = [self.y0]

        # loop over number of samples and integrate 1 time step each with dt length
        for i in range(self.samples):

            # start each time from 0 because we change starting conditions each iteration, time dependent variables (input u or other parameters) are iterated via samples i
            t = [0, self.dt]
            
            # parameters get either unpacked into a tuple for that specific sample (time step) if they are a np.ndarray or are given as just one value

            args = tuple(
                    [inp[i,:] if isinstance(inp,np.ndarray) else inp for inp in self.inputs] + 
                    [par[i,:] if isinstance(par,np.ndarray) else par for par in self.parameters] 
                )
            result = odeint(self.func, self.y0, t, args= args)
            self.y0 = result[1]
            data.append(result[1])

        self.raw = np.array(data)
        self.add_noise()

    def transform(self):

        self.X = self.raw[:-1,:]
        self.y = self.raw[1:,:]

        for inp in self.inputs:
            if isinstance(inp, np.ndarray):
                u = inp
            else:
                u = np.array([[inp]*self.samples])
                u = u.reshape((-1,1))
            
            self.X = np.hstack((self.X, u))


    def add_noise(self):

        noise = self.rng.normal(0, 0.01, self.raw.shape)
        self.meas = self.raw + noise

if __name__ == "__main__":
    
    # pendulum diff. equation
    # y'' + (g/l) * sin(y) = 0
    # split into two first order diff. equations 
    # y[0] = y
    # y[1] = y'
    # --> y[0]' = y[1]
    # --> y[1]' = - (g/l) * sin(y[0])
    
    def func(y,t,u,g):
        l = 1
    
        res = [y[1], -(g/l) * np.sin(y[0]) + u[0] + u[1]]
        return res
    
    # [angle'_init, angle_init]
    y0 = [0, 0.1]
    samples = 500
    dt = 0.01

    # if parameters should vary in time, give them as lists like this for [[par1(t=0),par2(t=0)], [par1(t=1),par2(t=1)], ... ]; just append them as list, then transpose
    # if single constant, just give them as one value
    u = []
    u.append([0]*samples)
    u.append([1]*samples)
    u = np.array(u)
    u = u.T
    g = 9.81
    
    # give input, then parameters (both as tuples); inputs = things the RNN/model gets as input as well, parameters = things the model is supposed to learn (potential changes in the system)
    x = Gen(func,(u,),(g,),y0,dt,samples)
    x.generate()
    x.transform()

    t = np.arange(0, samples*dt, dt)
    plt.plot(t,x.X)
    plt.show()
