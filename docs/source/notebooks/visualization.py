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
def plot_matrix_multiplication_as_heatmap(A: pt.Tensor,B: pt.Tensor,C: pt.Tensor, annotate: bool=True):
    n_rows, n_cols = A.size()
    n_rows, n_cols = B.size()
    height = MATRIX_WIDTH * n_rows / n_cols
    fig, ((ax1),(ax2),(ax3)) = plt.subplots(1, 3, figsize=(3*MATRIX_WIDTH, height))
    ax1.imshow(A)
    ax2.imshow(B)
    ax3.imshow(C)
    if annotate:
        for row in range(n_rows):
            for col in range(n_cols):
                ax1.text(row, col, "{:2.2f}".format(A[col, row]),
                        ha="center", va="center", color="w", fontsize=ANNO_SIZE)
                ax2.text(row, col, "{:2.2f}".format(B[col, row]),
                        ha="center", va="center", color="w", fontsize=ANNO_SIZE)
                ax3.text(row, col, "{:2.2f}".format(C[col, row]),
                        ha="center", va="center", color="w", fontsize=ANNO_SIZE)
    ax1.set_xticks(pt.arange(n_cols+1) - 0.5, minor=True)
    ax1.set_yticks(pt.arange(n_rows+1) - 0.5, minor=True)
    ax1.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax1.tick_params(which="minor", bottom=False, left=False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
    ax1.set_ylabel("{:d} rows".format(n_rows), fontsize=LABEL_SIZE)
    ax2.set_xticks(pt.arange(n_cols+1) - 0.5, minor=True)
    ax2.set_yticks(pt.arange(n_rows+1) - 0.5, minor=True)
    ax2.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax2.tick_params(which="minor", bottom=False, left=False)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
    ax2.set_ylabel("{:d} rows".format(n_rows), fontsize=LABEL_SIZE)
    ax3.set_xticks(pt.arange(n_cols+1) - 0.5, minor=True)
    ax3.set_yticks(pt.arange(n_rows+1) - 0.5, minor=True)
    ax3.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax3.tick_params(which="minor", bottom=False, left=False)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
    ax3.set_ylabel("{:d} rows".format(n_rows), fontsize=LABEL_SIZE)

