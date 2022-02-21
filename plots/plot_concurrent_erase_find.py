from email.policy import default
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import scipy.stats
from matplotlib.offsetbox import AnchoredText


def print_summary(label, insert_col, find_col):
    mean_insertion_rate = scipy.stats.hmean(insert_col.dropna())
    mean_find_rate = scipy.stats.hmean(find_col.dropna())
    print("{0: <25}".format(label) + ' | ',
          "{0: <15}".format(round(mean_insertion_rate, 3)) + '|',
          "{0: <8}".format(round(mean_find_rate, 3)))


def plot_rates(results_dir, output_dir, find_limits, spacing=[1, 50]):

    svg_name = ''
    subdir = '/versioned_find_erase/'

    colors = ['#1b9e77', '#d95f02', '#7570b3']
    update_ratios = [5, 50, 90]
    initial_sizes = [45]
    dfs = []
    for s in initial_sizes:
        p0 = 'rates_initial' + str(s) + 'M_'
        for u in update_ratios:
            p1 = 'update' + str(u)
            df = pd.read_csv(results_dir + subdir + p0 + p1 + '.csv')
            dfs.append(df)

    svg_name = 'erase_find_rates'
    # allocator_name = 'bump'
    allocator_name = 'slab'
    svg_name = svg_name + '_' + allocator_name

    print("{0: <25}".format('') + ' | ',
          "{0: <15}".format('BLink') + '|',
          "{0: <8}".format('VBTree'))

    for s in range(len(initial_sizes)):
        # use milions of keys as x axis
        df_offset = s * len(update_ratios)
        x_axis = dfs[df_offset]['num_updates'] + \
            dfs[df_offset]['num_queries']
        x_axis = x_axis.divide(1.0e6)

        font_size = 30  # font size
        scale = 10
        fig = plt.figure(figsize=(2*scale, 2*scale))
        ax = fig.add_subplot(111)

        subplots = []
        titles = ['']
        subplots.append(fig.add_subplot(1, 1, 1))
        subplots[-1].set_title(titles[0], fontweight="bold", size=font_size)

        # remove borders
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False,
                       bottom=False, left=False, right=False)

        ax.set_xlabel('Millions of Operations', fontsize=font_size)

        markers = ['s', 'o', '^', 'D']
        linewidth = 4

        column_names = ['vblink_' + allocator_name +
                        '_out_of_place_', 'blink_' + allocator_name + '_']
        titles = ['']
        for t in range(len(titles)):
            for u in range(len(update_ratios)):
                l = str(update_ratios[u]) + '% update' if t == 0 else ''
                subplots[t].plot(x_axis, dfs[df_offset + u + t * len(update_ratios)][column_names[0] + 'concurrent_ops'],
                                 marker=markers[u], label=l, linewidth=linewidth, color=colors[u])
                print_summary(str(initial_sizes[s]) + 'Mkeys_' + str(update_ratios[u]) + 'alpha', dfs[df_offset + u + t * len(update_ratios)][column_names[1] + 'concurrent_ops'],
                              dfs[df_offset + u + t * len(update_ratios)][column_names[0] + 'concurrent_ops'])

                subplots[t].plot(x_axis, dfs[df_offset + u + t * len(update_ratios)][column_names[1] + 'concurrent_ops'],
                                 marker=markers[u], label='', linewidth=linewidth, linestyle='dashed', color=colors[u])

            subplots[t].set_ylabel(
                'Operations Rate (MOp/s)', fontsize=font_size)

        for p in subplots:
            p.spines["right"].set_visible(False)
            p.spines["top"].set_visible(False)

        if find_limits != [-1, -1]:
            subplots[0].set_ylim(find_limits)

        for ax in fig.get_axes():
            ax.tick_params(axis='x', labelsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size)

        fig.tight_layout()

        fig.show()

        textstr = '\n'.join((r'╌╌╌  B-Tree', r'―― VoB-Tree'))
        fig.text(0.6, 0.14, textstr, fontsize=font_size)

        fig.legend(prop={'size': font_size},
                   bbox_to_anchor=(1, 0.2), frameon=False)

        fig.savefig(output_dir + '/' + svg_name +
                    str(initial_sizes[s]) + '.svg', bbox_inches='tight')
        fig.savefig(output_dir + '/' + svg_name +
                    str(initial_sizes[s]) + '.pdf', bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='./')
    parser.add_argument('-od', '--output-dir', default='./')
    parser.add_argument('-mf', '--min-ops-throughput', default=-1, type=int)
    parser.add_argument('-xf', '--max-ops-throughput', default=-1, type=int)

    plt.rcParams["image.cmap"] = "Set1"
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

    args = parser.parse_args()
    print("Reading results from: ", args.dir)
    plot_rates(args.dir, args.output_dir, [
               args.min_ops_throughput, args.max_ops_throughput], [1, 100])
