import numpy as np
from glob import glob
import argparse
import json
import matplotlib.pyplot as plt


def plot_avg_cost_graph(costs, colors, labels, title='Title', show=True, save=True,
                        path=None):
    fig, axs = plt.subplots(1, figsize=(10, 6))
    x = np.arange(len(costs[0])).tolist()

    for cost, color, label in zip(costs, colors, labels):
        axs.plot(x, cost, color=color, label=label)
    axs.set(title=title)
    axs.set(ylabel='Avg. Cost')
    axs.set(xlabel='Episode')
    axs.legend(loc='upper right')

    if save:
        plt.savefig(path + "avg_cost_plot.png")

    if show:
        plt.show(block=False)
        input()


def plot_dropped_ratio_graph(dropped_ratios, x_label, title='Title', show=True, save=True,
                             path=None):
    fig, axs = plt.subplots(1, figsize=(10, 6))

    dropped_ratios = np.array(sorted(dropped_ratios))

    axs.plot(dropped_ratios[:, 0], dropped_ratios[:, 1], color='green', label='DRL')
    axs.set(title=title)
    axs.set(ylabel='Dropper Tast Ratio')
    axs.set(xlabel=x_label)
    axs.legend(loc='upper right')

    if save:
        plt.savefig(path + "dropped_ratio_plot.png")

    if show:
        plt.show(block=False)
        input()


def plot_avg_delay_graph(avg_delay, x_label, title='Title', show=True, save=True,
                         path=None):
    fig, axs = plt.subplots(1, figsize=(10, 6))

    avg_delay = np.array(sorted(avg_delay))

    axs.plot(avg_delay[:, 0], avg_delay[:, 1], color='green', label='DRL')
    axs.set(title=title)
    axs.set(ylabel='Avg. Delay (Sec)')
    axs.set(xlabel=x_label)
    axs.legend(loc='upper right')

    if save:
        plt.savefig(path + "avg_delay_plot.png")

    if show:
        plt.show(block=False)
        input()


def main(args):
    dirs = glob(f"{args.path}/*/")

    if args.type == 'cost':
        costs = list()
        colors = list()
        labels = list()
        for dir in dirs:
            avg_costs_np = np.load(dir + "/plots/avg_cost.npy")
            avg_costs_np = np.convolve(avg_costs_np, np.ones((args.window,))/args.window,
                                       mode='valid')
            costs.append(avg_costs_np)
            with open(dir + "/plots/plot_props.dat") as fp:
                data = json.load(fp)
            colors.append(data['color'])
            labels.append(data['label'])
        plot_avg_cost_graph(costs, colors, labels, args.title, path=args.path)
    elif args.type == 'dropped':
        dropped_ratios = list()
        for dir in dirs:
            with open(dir + "/results/results.dat") as fp:
                data = json.load(fp)
            dropped_ratios.append(data['avg_dropped'])tigonglege1
        plot_dropped_ratio_graph(dropped_ratios, args.x_label, args.title, path=args.path)
    elif args.type == 'delay':
        avg_delays = list()
        for dir in dirs:
            with open(dir + "/results/results.dat") as fp:
                data = json.load(fp)
            avg_delays.append(data['avg_delay'])
        plot_avg_delay_graph(avg_delays, args.x_label, args.title, path=args.path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot Results for Mobile Edge Computing')
    parser.add_argument('--type', type=str, default='cost',
                        help='plot type: {cost, dropped, delay} (default: cost)')
    parser.add_argument('--path', type=str, default=None,
                        help='path to results directory (default: None)')
    parser.add_argument('--window', type=int, default=50,
                        help='moving average window size (default: 50)')
    parser.add_argument('--x_label', type=str, default=None,
                        help='x_label for dropper task and avg. delay plots')
    parser.add_argument('--title', type=str, default='Title',
                        help='plot title (default: Title)')
    args = parser.parse_args()

    main(args)