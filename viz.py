import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap


def cond_dists_heatmap(dists: [[[float]]]):
    plt.figure(figsize=(24, 6))
    n_cols = 3
    n_rows = (len(dists) - 1) // n_cols + 1

    for i, dist in enumerate(dists):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.heatmap(dist, cmap='Blues', vmin=0, vmax=dists.max(), linewidth=.1, cbar=(i + 1) % n_cols == 0,
                    cbar_kws=dict(format=FuncFormatter(lambda z, pos: '{:.0%}'.format(z)), label='Probability', ))
        plt.gca().invert_yaxis()  # have bottom left be (0, 0)
        if i % n_cols != 0:
            plt.yticks([])
        else:
            plt.ylabel('Node B degree')

        plt.xlabel('Node A degree')
        plt.tick_params(axis='x', bottom=False)
        plt.tick_params(axis='y', left=False)

        plt.title(f'Step {i}')

    plt.suptitle('Connections Distribution')
    plt.show()


def univar_hists(dists: [[float]]):
    univar_dists = dists.sum(axis=1)  # assumes symmetry of each bivariate distribution
    for i, d in enumerate(univar_dists):
        plt.plot(d, label=i)
    plt.title('Univariate Degree Distribution')

    ax = plt.gca()
    ax.set_yticklabels(['{:.0%}'.format(y) for y in ax.get_yticks()])

    plt.xlabel('Degree')
    plt.ylabel('Nodes')
    plt.legend(title='Timestep')
    plt.show()


def cond_dists_hist():
    """
    degrees_range = np.arange(1, len(distr)+1)

    cond_distr = distr / degrees_range[:, np.newaxis]
    cond_distr /= cond_distr.sum()

    plt.figure(figsize=(8, 6))

    cumsums = np.cumsum(cond_distr, axis=1)
    degrees_range = np.arange(1, len(cond_distr)+1)

    for i in reversed(range(len(cond_distr))):
        plt.bar(degrees_range, cond_distr[:, i],
                bottom=cumsums[:, i-1] if i > 0 else None,
                label=i+1)

    ax = plt.gca()
    ax.set_xticks(degrees_range)
    ax.set_yticklabels(['{:.0%}'.format(y) for y in ax.get_yticks()])

    plt.xlabel('Degree')
    plt.ylabel('Neurons')

    plt.title('#Connections Distribution')
    plt.legend(title="Other's degree")

    plt.show()
    """


def masks_time_heatmaps(n_layers, masks):
    n_timesteps = len(masks)
    colors_list = [
        np.ones(3) * .97,  # light grey
        *sns.color_palette('Blues', n_timesteps),
    ]

    plt.figure(figsize=(24, 5))
    for layer in range(n_layers - 1):
        plt.subplot(1, n_layers - 1, layer + 1)
        plt.title(f'Layer {layer} → {layer+1}')

        timesteps_active = masks[layer].sum(axis=0)
        sns.heatmap(timesteps_active,
                    cmap=ListedColormap(colors_list), linewidth=.01, vmin=0,
                    cbar=layer == n_layers - 2,
                    cbar_kws=dict(ticks=range(n_timesteps + 1),
                                  label='Timesteps active'))
        plt.ylabel(f'L{layer} node')
        plt.xlabel(f'L{layer+1} node')

    plt.suptitle('Connections')
    plt.show()
