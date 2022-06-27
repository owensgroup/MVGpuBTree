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


def plot_rates(results_dir, output_dir, find_limits, insert_limits, spacing=[1, 50]):

    svg_name = ''
    subdir = '/blink_vs_versioned/'
    csv_fname = 'rates.csv'
    df = pd.DataFrame()
    df = pd.read_csv(results_dir + subdir + csv_fname)

    # use milions of keys as x axis
    df['num_keys'] = df['num_keys'].divide(1.0e6)
    colors = ['#ca0020', '#f4a582', '#0571b0']

    font_size = 30  # font size
    scale = 10
    allocator_names = ['slab']
    for allocator_name in allocator_names:
        fig = plt.figure(figsize=(2*scale, 2*scale))
        ax = fig.add_subplot(111)

        titles = ['Insertion rates', 'Find rates']

        subplots = []
        subplots.append(fig.add_subplot(2, 1, 1))
        # subplots[-1].set_title(titles[0], fontweight="bold", size=font_size)
        subplots.append(fig.add_subplot(2, 1, 2))
        # subplots[-1].set_title(titles[1], fontweight="bold", size=font_size)

        # remove borders
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False,
                       bottom=False, left=False, right=False)

        list(df.columns)

        markers = ['s', 'o', '^', 'D']
        linewidth = 4
        # allocator_name = 'bump'
        svg_name = 'insertion_find_rates'
        svg_name = svg_name + '_' + allocator_name
        column_names = ['blink_' + allocator_name + '_',
                        'vblink_' + allocator_name + '_in_place_',
                        'vblink_' + allocator_name + '_out_of_place_']
        titles = ['B-Tree', 'ViB-Tree', 'VoB-Tree']
        for m, c, t, o in zip(markers, column_names, titles, colors):
            subplots[0].plot(df['num_keys'], df[c + 'insert'],
                             marker=m, label=t, linewidth=linewidth, color=o)
            subplots[1].plot(df['num_keys'], df[c + 'find'],
                             marker=m, linewidth=linewidth, color=o)

            print_summary(t + '_' + allocator_name,
                          df[c + 'insert'], df[c + 'find'])

        ax.set_xlabel('Millions of keys', fontsize=font_size)
        subplots[0].set_ylabel('Insert Rate (MKey/s)', fontsize=font_size)
        subplots[1].set_ylabel('Find Rate (MKey/s) ', fontsize=font_size)

        for p in subplots:
            p.spines["right"].set_visible(False)
            p.spines["top"].set_visible(False)

        if insert_limits != [-1, -1]:
            subplots[0].set_ylim(insert_limits)
        if find_limits != [-1, -1]:
            subplots[1].set_ylim(find_limits)

        for ax in fig.get_axes():
            ax.tick_params(axis='x', labelsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size)

        fig.tight_layout()

        fig.show()
        fig.legend(prop={'size': font_size},
                   bbox_to_anchor=(1, 0.84), frameon=False)

        fig.savefig(output_dir + '/' + svg_name + '.svg', bbox_inches='tight')
        fig.savefig(output_dir + '/' + svg_name + '.pdf', bbox_inches='tight')


def plot_rates_blink(results_dir, output_dir, find_limits, insert_limits, spacing=[1, 50]):

    svg_name = ''
    subdir = '/blink/'
    csv_fname = 'rates.csv'
    df = pd.DataFrame()
    df = pd.read_csv(results_dir + subdir + csv_fname)
    svg_name = 'blink_insertion_find_rates'

    # use milions of keys as x axis
    df['num_keys'] = df['num_keys'].divide(1.0e6)

    font_size = 30  # font size
    scale = 10
    fig = plt.figure(figsize=(2*scale, 2*scale))
    ax = fig.add_subplot(111)

    titles = ['Insertion rates', 'Find rates']

    subplots = []
    subplots.append(fig.add_subplot(2, 1, 1))
    # subplots[-1].set_title(titles[0], fontweight="bold", size=font_size)
    subplots.append(fig.add_subplot(2, 1, 2))
    # subplots[-1].set_title(titles[1], fontweight="bold", size=font_size)

    # remove borders
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False,
                   bottom=False, left=False, right=False)

    list(df.columns)

    markers = ['s', 'o', '^', 'D']
    linewidth = 4
    # allocator_name = 'bump'
    allocator_name = 'slab'
    svg_name = svg_name + '_' + allocator_name

    column_names = ['blink_' + allocator_name + '_']
    titles = ['B-Tree']
    print(svg_name)
    for m, c, t in zip(markers, column_names, titles):
        subplots[0].plot(df['num_keys'], df[c + 'insert'],
                         marker=m, label=t, linewidth=linewidth)
        subplots[1].plot(df['num_keys'], df[c + 'find'],
                         marker=m, linewidth=linewidth)

        print_summary(t, df[c + 'insert'], df[c + 'find'])

    ax.set_xlabel('Millions of keys', fontsize=font_size)
    subplots[0].set_ylabel('Insert Rate (MKey/s)', fontsize=font_size)
    subplots[1].set_ylabel('Find Rate (MKey/s) ', fontsize=font_size)

    for p in subplots:
        p.spines["right"].set_visible(False)
        p.spines["top"].set_visible(False)

    if insert_limits != [-1, -1]:
        subplots[0].set_ylim(insert_limits)
    if find_limits != [-1, -1]:
        subplots[1].set_ylim(find_limits)

    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

    fig.tight_layout()

    fig.show()
    fig.legend(prop={'size': font_size},
               bbox_to_anchor=(1, 0.7), frameon=False)

    fig.savefig(output_dir + '/' + svg_name + '.svg', bbox_inches='tight')
    fig.savefig(output_dir + '/' + svg_name + '.pdf', bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='./')
    parser.add_argument('-od', '--output-dir', default='./')
    parser.add_argument('-mf', '--min-find-throughput', default=-1, type=int)
    parser.add_argument('-xf', '--max-find-throughput', default=-1, type=int)
    parser.add_argument('-mi', '--min-insert-throughput', default=-1, type=int)
    parser.add_argument('-xi', '--max-insert-throughput', default=-1, type=int)

    plt.rcParams["image.cmap"] = "Set1"
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

    args = parser.parse_args()
    print("Reading results from: ", args.dir)
    plot_rates(args.dir, args.output_dir, [args.min_find_throughput, args.max_find_throughput],
               [args.min_insert_throughput, args.max_insert_throughput], [1, 100])
    # plot_rates_blink(args.dir, args.output_dir, [args.min_find_throughput, args.max_find_throughput],
    #  [args.min_insert_throughput, args.max_insert_throughput], [1, 100])
