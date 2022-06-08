import numpy as np
import random

length = 4
sample = np.array(range(1,5))

n = 1000
X = np.empty((n, length-1, 1))
y = np.empty((n, 1, 1))

# print(sample)

for row in range(n):

    x = sample * random.randint(1,100)
    X[row,:,0] = x[:-1]
    y[row,:,0] = x[-1]

# print(X[:3])
# print(y[:3])


def check_linearity(fun):

    x = 3
    y = 7
    CON = 5

    check = 1

    if fun(x + y) == fun(x) + fun(y):
        print("additivity passed!")
    else:
        check = 0
    
    if fun(CON * x) == CON * fun(x):
        print("homogeneity passed!")
    else:
        check = 0
    
    if check:
        print(f"{fun} is linear!")
    else:
        print(f"{fun} is non-linear!")


def lin(x):
    return 2 * x

def nonlin(x):
    return x**2


check_linearity(lin)
check_linearity(nonlin)
