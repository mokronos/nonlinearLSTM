import matplotlib.pyplot as plt
from helper import pred_name, gt_name, error_name, norm_name


def vis_results(results, data_config, savepath, suffix):
    indices = results.index.unique(level="series")
    for i in indices:


        data = results.xs(i)
        data.plot()

        for var in data_config["outputs"]:
            data[f"{var}_error"] =(data[gt_name(var)] - data[pred_name(var)])**2

        pred = data[[pred_name(x) for x in data_config["outputs"]]]
        gt = data[[gt_name(x) for x in data_config["outputs"]]]
        error = data[[error_name(x) for x in data_config["outputs"]]]

        fig, ax = plt.subplots(2, constrained_layout = True)
        ax[0].plot(pred, label=[f"pred_{x}" for x in data_config["outputs"]])
        ax[0].plot(gt, "--", label=[f"gt_{x}" for x in data_config["outputs"]])
        ax[0].set_ylabel(f"{data_config['output_labels'][0]} in {data_config['output_units'][0]}")
        ax[0].set_xlabel("time in 0.01s steps")
        ax[0].legend()
        ax[1].plot(error, label=[f"squ_error_{x}" for x in data_config["outputs"]])
        ax[1].set_ylabel(r"$m^2/s^2$")
        ax[1].set_xlabel("time in 0.01s steps")
        ax[1].legend()
        plt.savefig(f"{savepath}/{suffix}{i}.pdf")
        plt.savefig(f"{savepath}/{suffix}{i}.png")
        plt.close
        plt.clf()

def vis_data(df, data_config, savepath, suffix):
    indices = df.index.unique(level="series")
    for i in indices:

        data = df.xs(i)

        info = [f"{key}: {value}" for key, value in data_config["constants"].items()]
        info = "; ".join(info)

        fig, ax1 = plt.subplots()
        fig.suptitle(info)

        ax2 = ax1.twinx()

        color1 = "g"
        color2 = "r"

        data[data_config["outputs"]].plot(ax=ax1, color = color1)
        data[data_config["inputs"]].plot(ax=ax2, color = color2)

        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)

        ax1.legend(data_config["output_labels"], loc="upper left")
        ax2.legend(data_config["input_labels"], loc="lower right")

        ax1.set_xlabel(r"time in $0.01 s$")
        ax1.set_ylabel(fr'{data_config["output_labels"][0]} in {data_config["output_units"][0]}', color = color1)
        ax2.set_ylabel(fr'{data_config["input_labels"][0]} in {data_config["input_units"][0]}', color = color2)

        plt.savefig(f"{savepath}/{suffix}{i}.pdf")
        plt.savefig(f"{savepath}/{suffix}{i}.png")
        plt.close
        plt.clf()

