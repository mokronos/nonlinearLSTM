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

y0 = [1, 1]
t = np.arange(0, 5, 0.01)

y1 = odeint(model,y0,t, tfirst=True)
y2 = solve_ivp(model, [0,5], y0, dense_output=True, method = "LSODA")
y2 = y2.sol(t)
y3 = solve_ivp(model, [0,5], y0, dense_output=True)
y3 = y3.sol(t)

plt.figure(1)
plt.plot(t,y2.T-y1)

plt.figure(2)
plt.plot(t,y3.T-y1)

plt.figure(3)
plt.plot(t,y3.T-y2.T)

plt.show()
