"""Module with visualizations functions.
"""

import torch as pt
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 160

MATRIX_WIDTH = 3.0
LABEL_SIZE = 8
ANNO_SIZE = 6

def plot_matrix_as_heatmap(matrix: pt.Tensor, annotate: bool=True):
    n_rows, n_cols = matrix.size()
    height = MATRIX_WIDTH * n_rows / n_cols
    fig, ax = plt.subplots(1, 1, figsize=(MATRIX_WIDTH, height))
    ax.imshow(matrix)
    if annotate:
        for row in range(n_rows):
            for col in range(n_cols):
                ax.text(row, col, "{:2.2f}".format(matrix[row, col]),
                        ha="center", va="center", color="w", fontsize=ANNO_SIZE)

    ax.set_xticks(pt.arange(n_cols+1) - 0.5, minor=True)
    ax.set_yticks(pt.arange(n_rows+1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
    ax.set_ylabel("{:d} rows".format(n_rows), fontsize=LABEL_SIZE)
