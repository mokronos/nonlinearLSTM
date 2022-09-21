import matplotlib.pyplot as plt
import pandas as pd
from helper import load_dataset


name = "thermal_simple"
df, config = load_dataset(name)

print(df)
for i, data in df.groupby(level=0):
    print(data.head())
    data.plot()
# df.loc[3].plot()
plt.show()
