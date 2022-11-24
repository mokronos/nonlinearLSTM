import matplotlib.pyplot as plt
from helper import c_one, pred_name, gt_name, error_name, norm_name


def vis_results_1(results, data_config, savepath, suffix):
    indices = results.index.unique(level="series")
    for i in indices:


        data = results.xs(i)

        for var in data_config["outputs"]:
            data[f"{var}_error"] =(data[gt_name(var)] - data[pred_name(var)])**2

        pred = data[[pred_name(x) for x in data_config["outputs"]]]
        gt = data[[gt_name(x) for x in data_config["outputs"]]]
        error = data[[error_name(x) for x in data_config["outputs"]]]

        fig, ax = plt.subplots(2, constrained_layout = True)

        cout0 = "b"
        cout0gt = "c--"
        cout1 = "r"
        cout1gt = "m--"
        cerror0 = "b"
        cerror1 = "r"
        cinp = "g"

        ax[0].plot(pred,cout0, label=c_one([fr"${x}_{{pred}}$" for x in data_config["output_labels"]]))
        ax[0].plot(gt, cout0gt, label=c_one([fr"${x}_{{gt}}$" for x in data_config["output_labels"]]))
        ax[0].set_ylabel(c_one([fr"${x}$ in ${y}$" for x, y in zip(data_config["output_labels"], data_config["output_units"])][0]))
        ax[0].set_xlabel("time in 0.01s steps")
        ax[0].legend()

        ax[1].plot(error, cerror0, label=c_one([fr"${x}_{{error}}$" for x in data_config["output_labels"]]))
        err_unit = data_config["error_units"][0]
        out_label = data_config["output_labels"][0]
        ax[1].set_ylabel(fr"${out_label}_{{error}}$ in ${err_unit}$")
        ax[1].set_xlabel("time in 0.01s steps")
        ax[1].legend()


        plt.savefig(f"{savepath}/{suffix}{i}.pdf")
        plt.savefig(f"{savepath}/{suffix}{i}.png")
        plt.close()
        plt.clf()

def vis_results_2(results, data_config, savepath, suffix):
    indices = results.index.unique(level="series")
    for i in indices:


        data = results.xs(i)

        for var in data_config["outputs"]:
            data[f"{var}_error"] =(data[gt_name(var)] - data[pred_name(var)])**2

        pred = data[[pred_name(x) for x in data_config["outputs"]]]
        gt = data[[gt_name(x) for x in data_config["outputs"]]]
        error = data[[error_name(x) for x in data_config["outputs"]]]

        fig, ax = plt.subplots(2)
        plt.tight_layout()
        fig.subplots_adjust(left=0.25)

        ax11 = ax[0]
        ax12 = ax[0].twinx()
        ax21 = ax[1]
        ax22 = ax[1].twinx()

        ax12.spines.left.set_position(("axes", -0.2))
        ax12.yaxis.set_label_position("left")
        ax12.yaxis.set_ticks_position("left")
        ax22.spines.left.set_position(("axes", -0.2))
        ax22.yaxis.set_label_position("left")
        ax22.yaxis.set_ticks_position("left")

        
        pred0 = pred[pred_name(data_config["outputs"][0])]
        pred1 = pred[pred_name(data_config["outputs"][1])]
        gt0 = gt[gt_name(data_config["outputs"][0])]
        gt1 = gt[gt_name(data_config["outputs"][1])]
        error0 = error[error_name(data_config["outputs"][0])]
        error1 = error[error_name(data_config["outputs"][1])]
        label0 = data_config["output_labels"][0]
        label1 = data_config["output_labels"][1]
        unit0 = data_config["output_units"][0]
        unit1 = data_config["output_units"][1]
        uniterr0 = data_config["error_units"][0]
        uniterr1 = data_config["error_units"][1]


        cout0 = "b"
        cout0gt = "c--"
        cout1 = "r"
        cout1gt = "m--"
        cerror0 = "b"
        cerror1 = "r"
        cinp = "g"

        c11 = cout0
        c11_gt = cout0gt
        c12 = cout1
        c12_gt = cout1gt

        c21 = cerror0
        c22 = cerror1
        fontsize = 6
        
        p1, = ax11.plot(pred0, c11, label=fr"${label0}_{{pred}}$")
        p2, = ax11.plot(gt0, c11_gt, label=fr"${label0}_{{gt}}$")
        ax11.set_ylabel(fr"${label0}$ in ${unit0}$")

        p3, = ax12.plot(pred1, c12, label=fr"${label1}_{{pred}}$")
        p4, = ax12.plot(gt1, c12_gt, label=fr"${label1}_{{gt}}$")
        ax12.set_ylabel(fr"${label1}$ in ${unit1}$")

        ax11.set_xlabel("time in 0.01s steps")
        ax11.legend(handles=[p1, p2, p3, p4], fontsize=fontsize)

        p5, = ax21.plot(error0, c21, label=fr"${label0}_{{error}}$")
        ax21.set_ylabel(fr"${label0}_{{error}}$ in ${uniterr0}$")

        p6, = ax22.plot(error1, c22, label=fr"${label1}_{{error}}$")
        ax22.set_ylabel(fr"${label1}_{{error}}$ in ${uniterr1}$")

        ax21.set_xlabel("time in 0.01s steps")

        ax11.yaxis.label.set_color(p1.get_color())
        ax12.yaxis.label.set_color(p3.get_color())
        ax21.yaxis.label.set_color(p5.get_color())
        ax22.yaxis.label.set_color(p6.get_color())

        tkw = dict(size=4, width=1.5)
        ax11.tick_params(axis='y', colors=p1.get_color(), **tkw)
        ax12.tick_params(axis='y', colors=p3.get_color(), **tkw)
        ax21.tick_params(axis='y', colors=p5.get_color(), **tkw)
        ax22.tick_params(axis='y', colors=p6.get_color(), **tkw)
        ax11.tick_params(axis='x', **tkw)
        ax21.tick_params(axis='x', **tkw)

        ax21.legend(handles=[p5, p6], fontsize=fontsize)


        plt.tight_layout()
        plt.savefig(f"{savepath}/{suffix}{i}.pdf")
        plt.savefig(f"{savepath}/{suffix}{i}.png")
        plt.close()
        plt.clf()


def vis_data_1(df, data_config, savepath, suffix):
    indices = df.index.unique(level="series")
    for i in indices:

        data = df.xs(i)

        info = [f"{key}: {value}" for key, value in data_config["constants"].items()]
        info = "; ".join(info)

        fig, ax1 = plt.subplots()

        # title with constants of dataset
        # fig.suptitle(info)

        ax2 = ax1.twinx()

        cout0 = "b"
        cout0gt = "c--"
        cout1 = "r"
        cout1gt = "m--"
        cerror0 = "b"
        cerror1 = "r"
        cinp = "g"

        outlabel = data_config["output_labels"][0]
        inlabel = data_config["input_labels"][0]

        p1, = ax1.plot(data[data_config["outputs"]], color = cout0, label = f"${outlabel}$")
        p2, = ax2.plot(data[data_config["inputs"]], color = cinp, label = f"${inlabel}$")

        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)

        ax1.set_xlabel(r"time in $0.01 s$")
        ax1.set_ylabel(c_one([f"${x}$ in ${y}$" for x, y in zip(data_config["output_labels"], data_config["output_units"])]))
        ax2.set_ylabel(c_one([f"${x}$ in ${y}$" for x, y in zip(data_config["input_labels"], data_config["input_units"])]))

        ax1.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())

        tkw = dict(size=4, width=1.5)
        ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
        ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
        ax1.tick_params(axis='x', **tkw)

        ax1.legend(handles=[p1, p2])

        plt.tight_layout()
        plt.savefig(f"{savepath}/{suffix}{i}.pdf")
        plt.savefig(f"{savepath}/{suffix}{i}.png")
        plt.close()
        plt.clf()

def vis_data_2(df, data_config, savepath, suffix):
    indices = df.index.unique(level="series")
    for i in indices:

        data = df.xs(i)

        info = [f"{key}: {value}" for key, value in data_config["constants"].items()]
        info = "; ".join(info)

        fig, ax1 = plt.subplots()
        fig.subplots_adjust(left=0.25)

        # title with constants of dataset
        # fig.suptitle(info)

        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        ax2.spines.left.set_position(("axes", -0.2))
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.set_ticks_position("left")

        cout0 = "b"
        cout0gt = "c--"
        cout1 = "r"
        cout1gt = "m--"
        cerror0 = "b"
        cerror1 = "r"
        cinp = "g"


        p1, = ax1.plot(data[data_config["outputs"][0]],cout0, label=f"${data_config['output_labels'][0]}$")
        p2, = ax2.plot(data[data_config["outputs"][1]],cout1, label=f"${data_config['output_labels'][1]}$")
        p3, = ax3.plot(data[data_config["inputs"]],cinp, label=f"${c_one(data_config['input_labels'])}$")

        ax1.set_xlabel(r"time in $0.01 s$")
        y1, y2 = c_one([f"${x}$ in ${y}$" for x, y in zip(data_config["output_labels"], data_config["output_units"])])
        y3 = c_one([f"${x}$ in ${y}$" for x, y in zip(data_config["input_labels"], data_config["input_units"])])

        ax1.set_ylabel(y1)
        ax2.set_ylabel(y2)
        ax3.set_ylabel(y3)

        ax1.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
        ax3.yaxis.label.set_color(p3.get_color())

        tkw = dict(size=4, width=1.5)
        ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
        ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
        ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
        ax1.tick_params(axis='x', **tkw)
        ax1.legend(handles=[p1,p2,p3])

        plt.tight_layout()
        plt.savefig(f"{savepath}/{suffix}{i}.pdf")
        plt.savefig(f"{savepath}/{suffix}{i}.png")
        plt.close()
        plt.clf()

