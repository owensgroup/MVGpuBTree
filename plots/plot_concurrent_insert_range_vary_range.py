from email.policy import default
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import scipy.stats
from matplotlib.offsetbox import AnchoredText
from os import listdir
from os.path import isfile, join, splitext


def plot_rates(results_dir, output_dir, memory_limits, fname, spacing=[1, 50]):

    svg_name = ''
    subdir = '/versioned_insert_range_variable_range/'

    colors = ['#1b9e77', '#d95f02', '#7570b3']
    range_lengths = [8]
    update_ratios = [50]
    initial_sizes = [40]
    dfs = []
    df = pd.read_csv(results_dir + subdir + fname + '.csv')
    dfs.append(df)

    svg_name = 'insertion_vary_rq_' + fname
    # allocator_name = 'bump'
    allocator_name = 'slab'
    svg_name = svg_name + '_' + allocator_name

    # print("{0: <25}".format('') + ' | ',
    #       "{0: <15}".format('Reclaim') + '|',
    #       "{0: <8}".format('No-Reclaim'))

    if True:
        # use milions of keys as x axis
        x_axis = dfs[0]['average_range_length']

        font_size = 60  # font size
        scale = 10
        fig = plt.figure(figsize=(2*scale, 2*scale))
        ax = fig.add_subplot(111)

        # titles = ['Range length = 8', 'Range length = 32']

        subplots = []
        subplots.append(fig.add_subplot(1, 1, 1))
        # subplots[-1].set_title(titles[0], fontweight="bold", size=font_size)

        # remove borders
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False,
                       bottom=False, left=False, right=False)

        ax.set_xlabel('Average Range Length', fontsize=font_size)

        markers = ['s', 'o', '^', 'D']
        linewidth = 4

        column_names = ['vblink_slab_out_of_place_concurrent_ops',
                        'blink_slab_concurrent_ops']
        subplots[0].plot(x_axis, dfs[0][column_names[0]],
                         marker=markers[0], label='VoB-Tree', linewidth=linewidth, color=colors[0])
        subplots[0].plot(x_axis, dfs[0][column_names[1]],
                         marker=markers[1], label='B-Tree', linewidth=linewidth, color=colors[1])

        subplots[0].set_ylabel(
            'Operations Rate (MOp/s)', fontsize=font_size)

        for p in subplots:
            p.spines["right"].set_visible(False)
            p.spines["top"].set_visible(False)

        if memory_limits != [-1, -1]:
            subplots[0].set_ylim(memory_limits)

        for ax in fig.get_axes():
            ax.tick_params(axis='x', labelsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size)

        fig.tight_layout()

        fig.show()

        fig.legend(prop={'size': font_size},
                   bbox_to_anchor=(1, 1), frameon=False)

        fig.savefig(output_dir + '/' + svg_name + '.svg', bbox_inches='tight')
        fig.savefig(output_dir + '/' + svg_name + '.pdf', bbox_inches='tight')
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='./')
    parser.add_argument('-od', '--output-dir', default='./')
    parser.add_argument('-mf', '--min-ops', default=-1, type=int)
    parser.add_argument('-xf', '--max-ops', default=-1, type=int)

    args = parser.parse_args()
    print("Reading results from: ", args.dir)
    subdir = '/versioned_insert_range_variable_range/'

    files = [f for f in listdir(args.dir + subdir)
             if isfile(join(args.dir + subdir, f))]
    for f in files:
        if "num_ops" in f:
            plot_rates(args.dir, args.output_dir, [
                args.min_ops, args.max_ops], splitext(f)[0], [1, 100])
