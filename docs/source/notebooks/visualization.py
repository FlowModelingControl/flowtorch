"""Module with visualizations functions.
"""

import torch as pt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

mpl.rcParams['figure.dpi'] = 160

CELL_WIDTH = 0.75
LABEL_SIZE = 8
ANNO_SIZE = 6


def select_font_color(value, mean_value):
    if value < mean_value:
        return "w"
    else:
        return "k"


def matrix_shape(matrix):
    if len(matrix.size()) < 2:
        return matrix.size(), 1
    else:
        return matrix.size()


def get_labels(matrix):
    rows, cols = matrix_shape(matrix)
    if rows == 1:
        ylabel = "{:1d} row".format(rows)
    else:
        ylabel = "{:1d} rows".format(rows)
    if cols == 1:
        xlabel = "{:1d} column".format(cols)
    else:
        xlabel = "{:1d} columns".format(cols)
    return xlabel, ylabel


def annotate_matrix(axis, matrix):
    rows, cols = matrix_shape(matrix)
    mean = pt.mean(matrix.type(pt.float32))
    for row in range(rows):
        for col in range(cols):
            axis.text(
                col, row, "{:2.2f}".format(matrix[row, col]),
                ha="center", va="center", fontsize=ANNO_SIZE,
                color=select_font_color(matrix[row, col], mean)
            )


def create_canvas(matrices):
    total_rows = sum([matrix_shape(matrix)[1] for matrix in matrices])
    total_cols = sum([matrix_shape(matrix)[1] for matrix in matrices])
    fig = plt.figure(figsize=(total_cols*CELL_WIDTH, total_rows*CELL_WIDTH), constrained_layout=True)
    ratios = [matrix_shape(matrix)[1] for matrix in matrices]
    spec = gridspec.GridSpec(
        ncols=len(matrices), nrows=1, width_ratios=ratios,
        height_ratios=[1], figure=fig
    )
    axarr = [fig.add_subplot(spec[0, i]) for i in range(len(matrices))]
    return fig, axarr


def plot_matrices_as_heatmap(matrices: list, titles: list = None, annotate: bool = True):
    fig, axarr = create_canvas(matrices)
    for matrix, ax in zip(matrices, axarr):
        ax.imshow(matrix)
        xlabel, ylabel = get_labels(matrix)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        rows, cols = matrix_shape(matrix)
        ax.set_xticks(pt.arange(cols+1) - 0.5, minor=True)
        ax.set_yticks(pt.arange(rows+1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        if annotate:
            annotate_matrix(ax, matrix)
    if titles is not None:
        assert len(titles) == len(axarr)
        for ax, title in zip(axarr, titles):
            ax.set_title(title, fontsize=LABEL_SIZE)