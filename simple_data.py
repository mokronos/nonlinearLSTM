import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "data/"
ext = ".pkl"
name = "power1.1"

f = lambda x: x**1.1

t = np.arange(1000)

y = f(t)
raw = pd.DataFrame({"t": t, "y": y})

pd.to_pickle(raw, f"{data_path}{name}{ext}")
