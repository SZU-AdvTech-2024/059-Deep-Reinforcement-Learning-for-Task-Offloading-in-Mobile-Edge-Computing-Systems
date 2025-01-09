import numpy as np
import matplotlib.pyplot as plt


def plot_graphs(axs, train_cost, train_dropped, train_delay, show=False, save=False,
                path=None):
    x = np.arange(len(train_cost)).tolist()
    axs[0].clear()
    axs[0].plot(x, train_cost, color='red', label='Training')
    axs[0].set(title='Avg. Cost')
    axs[0].set(ylabel='Avg. Cost')
    axs[0].set(xlabel='Episode')
    axs[0].legend(loc='upper right')

    axs[1].clear()
    axs[1].plot(x, train_dropped, color='blue', label='Training')
    axs[1].set(title='Ratio of Dropped Tasks')
    axs[1].set(ylabel='Dropped Ratio')
    axs[1].set(xlabel='Episode')
    axs[1].legend(loc='upper right')

    axs[2].clear()
    axs[2].plot(x, train_delay, color='green', label='Training')
    axs[2].set(title='Avg. Task Delay')
    axs[2].set(ylabel='Avg. Delay (Sec)')
    axs[2].set(xlabel='Episode')
    axs[2].legend(loc='upper right')

    if save:
        plt.savefig(path + "plots/learning_curves.png")

        with open(path + 'plots/avg_cost.npy', 'wb') as f:
            np.save(f, np.array(train_cost))

        with open(path + 'plots/dropped_ratio.npy', 'wb') as f:
            np.save(f, np.array(train_dropped))

        with open(path + 'plots/avg_delay.npy', 'wb') as f:
            np.save(f, np.array(train_delay))

    if show:
        plt.show(block=False)
        plt.pause(0.01)