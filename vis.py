import matplotlib.pyplot as plt
import pandas as pd


def plot_results(data, labels):
    
    plt.plot(data)
    plt.show()


x = 3

savepath = "data/"

# df = pd.read_pickle(savepath + "pendulum.pkl")
df = pd.read_pickle("data/pendulum.pkl")
df.head()
print(df["angle"])
