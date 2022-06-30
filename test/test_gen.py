import numpy as np
from ..ode import Gen

def test_gen():

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

    y0 = [0, 0.1]
    samples = 500
    dt = 0.01

    x = Gen(model,None,y0,dt,samples)
    x.generate()
    raw = [round(x,2) for x in x.raw[0]]

    assert raw == [0, 0.1]
