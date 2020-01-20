from collections import defaultdict
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu


def load_groups(files):
    """
    Loads all files into memory and organizes them into groups by their args objects.

    :param files: list of filenames
    :return: dictionary mapping a representative args object to a list of (filename, data) tuples
    """
    groups = defaultdict(list)
    for f in files:
        d = np.load(f, allow_pickle=True)
        gkey = to_group_key(d['args'].item()._get_kwargs())
        groups[gkey].append((f, d))
    return groups


def to_group_key(args_items):
    """
    Removes the seed and uid entries from the args_items object.

    :param args_items: list of (key, value) tuples
    :return: same list without the seed and uid keys
    """
    args_items = dict(args_items)
    del args_items['seed']
    del args_items['uid']
    return tuple(args_items.items())


def get_series(gval, series):
    """
    Returns an array of shape [group_size, timesteps] that contains the series values.

    :param gval: list of (file, data) objects
    :param series: name of the 1-d series that will be extracted
    """
    minlen = min([len(d[series]) for f, d in gval])
    return np.stack([d[series][:minlen] for f, d in gval])


def format_group_key(gkey):
    """
    Returns a formatted string that contains the important parts of the given group key.
    """
    args = dict(gkey)
    del args['dataset']
    del args['dataroot']
    del args['device']
    del args['iterations']
    del args['out']
    del args['test_interval']
    replacements = {
        'num_tasks': 'T',
        'num_slots': 'S',
        'num_modules': 'K',
        'batch_size': 'bs',
        'sgd_lr': r'$\eta_\theta$',
        'sgd_samples': r'$\lambda_\theta$',
        'nes_lr': r'$\eta_\pi$',
        'nes_samples': r'$\lambda_\pi$',
    }
    out = ''
    for k, v in args.items():
        if k in replacements:
            k = replacements[k]
        out += f'{k}={v}, '
    return out[:-2]


def plot_series(groups, series):
    """
    Plots the specified series. A lineplot is used to display the series' mean and a
    shaded area around it defines one standard deviation.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(series)

    for gkey, gval in groups.items():
        args = dict(gkey)

        series_values = get_series(gval, series)
        interval_size = args['test_interval']
        interval_count = series_values.shape[1] - 1

        x = np.arange(0, interval_size * interval_count + 1, step=interval_size)
        mean = np.mean(series_values, axis=0)
        std = np.std(series_values, axis=0)

        ax.plot(x, mean, label=format_group_key(gkey))
        ax.fill_between(x, mean + std, mean - std, alpha=0.2)

    ax.legend()
    return fig, ax


def plot_average_sharing_configuration(gval):
    """
    Plots the sharing configuration as a stacked bar plot.

    The x-axis defines the network position and each stacked bar at an x-position is made up of one bar
    for each possible sharing group size, i.e. from 1 (when a module is specific to a single task) to
    T (when a module is shared between all tasks). Each bar part's size is the average number of modules
    belonging to a group of that specific size over all experiments.
    """
    group_count_avg = 0
    for f, d in gval:
        # routing_probs is a 3-d array with dimensions [position, task, module]
        routing_probs = d['test/routing_probs'][-1]
        num_positions = routing_probs.shape[0]
        num_tasks = routing_probs.shape[1]
        num_modules = routing_probs.shape[2]
        # modules is a 2-d array with dimensions [position, task] that contains indices into the
        # modules at each position in the network
        modules = np.argmax(routing_probs, axis=-1)
        # For each position in the network separately: Check for each module how many tasks use it,
        # then bincount the result.
        # usage_count[pos, i] contains how many tasks use module i at position pos
        usage_count = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_modules + 1), axis=1, arr=modules)
        # group_count[pos, j] contains how many groups of modules that are shared between
        # exactly j tasks exist at position pos
        group_count = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_tasks + 1), axis=1, arr=usage_count)
        # Convert to a number of modules
        # Add to total which is of dimensions [pos, j]
        group_count_avg += group_count * np.linspace(0, 1, num_tasks + 1) / len(gval)

    max_group_size = np.max(np.nonzero(np.sum(group_count_avg, axis=0)))
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=1, vmax=max_group_size)

    x = list(range(1, num_positions + 1))
    y = 0

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.set_xlabel("Position in network")
    ax.set_ylabel("Percentage of tasks\nsharing weights with $t$ tasks")
    ax.set_xticks(x)

    for j in range(1, max_group_size + 1):
        group_j = 100 * group_count_avg[:, j]
        ax.bar(x, group_j, bottom=y, color=cmap(norm(j)))
        y += group_j

    # Create custom colorbar instead of legend
    scalar_map = mpl.cm.ScalarMappable(norm, cmap)
    cbar = fig.colorbar(scalar_map, ticks=range(1, max_group_size + 1))
    cbar.set_label('Group size $t$', rotation=90, labelpad=10)

    fig.tight_layout(rect=(-0.03, -0.05, 1.05, 1.05))
    return fig, ax


def plot_single_sharing_configuration(gval, index=0):
    """
    Plots a single sharing configuration (by default of the first experiment in gval).
    """
    routing_probs = gval[index][1]['test/routing_probs'][-1]
    modules = np.argmax(routing_probs, axis=-1)
    num_pos = modules.shape[0]
    num_tasks = modules.shape[1]
    num_modules = np.max(modules)

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=num_modules)

    fig, ax = plt.subplots()
    ax.set_xlabel("Position in network")
    ax.set_ylabel("Task")
    ax.set_xticks(range(1, num_pos + 1))
    ax.set_yticks(range(1, num_tasks + 1))
    ax.set_xlim(0.5, num_pos + 0.5)
    ax.set_ylim(0.5, num_tasks + 0.5)

    radius_outer = 0.25
    radius_inner = 0.15

    # Draw modules
    for pos in range(num_pos):
        for task in range(num_tasks):
            mod = modules[pos, task]
            ax.text(pos + 1, task + 1, f'{pos + 1}.{mod}', va='center', ha='center')
            circle1 = plt.Circle((pos + 1, task + 1), radius_outer, color=cmap(norm(mod)))
            circle2 = plt.Circle((pos + 1, task + 1), radius_inner, color='w')
            ax.add_artist(circle1)
            ax.add_artist(circle2)

    # Draw arrows between modules
    for pos in range(num_pos - 1):
        for task in range(num_tasks):
            ax.arrow(pos + 1 + radius_outer, task + 1, 1 - 2 * radius_outer - 0.1, 0,
                     head_width=0.05, head_length=0.1, fc='k', ec='k')


def print_groups(groups):
    for gkey, gval in groups.items():
        error = 1 - get_series(gval, 'test/metric')
        mean = np.mean(error, axis=0)[-1]
        std = np.std(error, axis=0)[-1]

        print(gkey)
        print("-----------")
        print(f"Test error: {100 * mean:.2f} +- {100 * std:.2f}")
        for f, d in gval:
            print(f, 1 - d['test/metric'][-1])
        print()


def is_error_significantly_smaller(groups, a, b):
    values_a = 1 - get_series(groups[a], 'test/metric')
    values_b = 1 - get_series(groups[b], 'test/metric')
    return mannwhitneyu(values_a[:, -1], values_b[:, -1], alternative='less')[-1]


def find_gkey(groups, query):
    """
    Returns the first matching gkey.

    :param groups: groups dictionary
    :param query: dictionary of gkey parameters and values
    """
    for gkey in groups.keys():
        args = dict(gkey)
        if all(args[k] == v for k, v in query.items()):
            return gkey


mnist = load_groups(glob('results/mnist*'))
cifar = load_groups(glob('results/cifar*'))
omniglot = load_groups(glob('results/omniglot*'))

print_groups(mnist)
print_groups(cifar)
print_groups(omniglot)


# Significance Tests
sigtests = [
    (mnist, 'mnist', 'learned_sharing', 'no_sharing'),
    (cifar, 'cifar', 'LearnedSharingResNet18', 'FullSharingResNet18'),
    (omniglot, 'omniglot', 'learned_sharing', 'full_sharing'),
    (omniglot, 'omniglot', 'LearnedSharingResNet18', 'FullSharingResNet18'),
    (omniglot, 'omniglot', 'LearnedSharingResNet18', 'NoSharingResNet18'),
    (omniglot, 'omniglot', 'FullSharingResNet18', 'LearnedSharingResNet18'),
]
for groups, name, a, b in sigtests:
    gkey_a = find_gkey(groups, {'model': a})
    gkey_b = find_gkey(groups, {'model': b})
    p = is_error_significantly_smaller(groups, gkey_a, gkey_b)
    print(f"{name}: {a} < {b} with pval = {p:.2e}")


# # Sharing Configuration Averages (Figures)
# figs1 = [
#     (mnist, 'mnist', 'learned_sharing'),
#     (cifar, 'cifar', 'learned_sharing'),
#     (cifar, 'cifar', 'LearnedSharingResNet18'),
#     (omniglot, 'omniglot', 'learned_sharing'),
#     (omniglot, 'omniglot', 'LearnedSharingResNet18'),
# ]
# for group, name, model in figs1:
#     gkey = find_gkey(group, {'model': model})
#     fig, ax = plot_average_sharing_configuration(group[gkey])
#     fig.savefig(f'fig-{name}-{model}.pdf')


# Sharing Configuration Single (Figure)
gkey = find_gkey(mnist, {'model': 'learned_sharing'})
plot_single_sharing_configuration(mnist[gkey], index=0)
plt.show()


# Metric line plots
plot_series(mnist, 'test/entropy')
plot_series(mnist, 'test/metric')
plt.show()
