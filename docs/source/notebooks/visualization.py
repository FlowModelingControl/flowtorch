"""Module with visualizations functions.
"""

import torch as pt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

mpl.rcParams['figure.dpi'] = 160

MATRIX_WIDTH = 3.0
LABEL_SIZE = 8
ANNO_SIZE = 6
Image_size = 160


def plot_matrix_as_heatmap(matrix: pt.Tensor, annotate: bool = True):
    n_rows, n_cols = matrix.size()
    height = MATRIX_WIDTH * n_rows / n_cols
    fig, ax = plt.subplots(1, 1, figsize=(MATRIX_WIDTH, height))
    ax.imshow(matrix)
    if annotate:
        for row in range(n_rows):
            for col in range(n_cols):
                ax.text(
                    row, col, "{:2.2f}".format(matrix[col, row]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )

    ax.set_xticks(pt.arange(n_cols+1) - 0.5, minor=True)
    ax.set_yticks(pt.arange(n_rows+1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
    ax.set_ylabel("{:d} rows".format(n_rows), fontsize=LABEL_SIZE)


def plot_2_matries_as_heatmap(A: pt.Tensor, B: pt.Tensor,
                              annotate: bool = True):
    n_rows, n_cols = A.size()
    n_rows, n_cols = B.size()
    height = MATRIX_WIDTH * n_rows / n_cols
    fig, axarr = plt.subplots(1, 2, figsize=(3*MATRIX_WIDTH, height))
    axarr[0].imshow(A)
    axarr[1].imshow(B)
    if annotate:
        for ax, mat in zip(axarr, [A, B]):
            for row in range(n_rows):
                for col in range(n_cols):
                    ax.text(
                        row, col, "{:2.2f}".format(mat[col, row]),
                        ha="center", va="center", color="w", fontsize=ANNO_SIZE
                    )

    for ax in axarr:
        ax.set_xticks(pt.arange(n_cols+1) - 0.5, minor=True)
        ax.set_yticks(pt.arange(n_rows+1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        axarr[0].set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
        axarr[0].set_ylabel("{:d} rows".format(n_rows), fontsize=LABEL_SIZE)
        axarr[1].set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
        axarr[1].set_ylabel(
            '+       j', fontsize=15, multialignment='center',
            rotation='horizontal', labelpad=28
        )


def plot_matrix_multiplication_as_heatmap(A: pt.Tensor, B: pt.Tensor,
                                          C: pt.Tensor, annotate: bool = True):
    n_rows, n_cols = A.size()
    n_rows, n_cols = B.size()
    n_rows, n_cols = C.size()
    height = MATRIX_WIDTH * n_rows / n_cols
    fig, axarr = plt.subplots(1, 3, figsize=(3*MATRIX_WIDTH, height))
    axarr[0].imshow(A)
    axarr[1].imshow(B)
    axarr[2].imshow(C)
    if annotate:
        for ax, mat in zip(axarr, [A, B, C]):
            for row in range(n_rows):
                for col in range(n_cols):
                    ax.text(
                        row, col, "{:2.2f}".format(mat[col, row]),
                        ha="center", va="center", color="w", fontsize=ANNO_SIZE
                    )

    for ax in axarr:
        ax.set_xticks(pt.arange(n_cols+1) - 0.5, minor=True)
        ax.set_yticks(pt.arange(n_rows+1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("{:d} columns".format(n_cols), fontsize=LABEL_SIZE)
        ax.set_ylabel("{:d} rows".format(n_rows), fontsize=LABEL_SIZE)


def plot_vector_multiplication_as_heatmap(A: pt.Tensor, B: pt.Tensor,
                                          C: pt.Tensor, annotate: bool = True):
    n_rowsA, n_colsA = A.size()
    n_rowsB, n_colsB = B.size()
    n_rowsC, n_colsC = C.size()
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(
        ncols=3, nrows=1, width_ratios=[n_colsA, n_colsB, n_colsC],
        height_ratios=[3], figure=fig
    )
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax1.imshow(A)
    ax2.imshow(B)
    ax3.imshow(C)
    if annotate:
        for row_A in range(n_rowsA):
            for col_A in range(n_colsA):
                ax1.text(
                    col_A, row_A, "{:2.2f}".format(A[row_A, col_A]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
        for row_B in range(n_rowsB):
            for col_B in range(n_colsB):
                ax2.text(
                    col_B, row_B, "{:2.2f}".format(B[row_B, col_B]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
        for row_C in range(n_rowsC):
            for col_C in range(n_colsC):
                ax3.text(
                    col_C, row_C, "{:2.2f}".format(C[row_C, col_C]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
    for ax, in zip([ax1, ax2, ax3]):
        ax1.set_xticks(pt.arange(n_colsA+1) - 0.5, minor=True)
        ax1.set_yticks(pt.arange(n_rowsA+1) - 0.5, minor=True)
        ax2.set_xticks(pt.arange(n_colsB+1) - 0.5, minor=True)
        ax2.set_yticks(pt.arange(n_rowsB+1) - 0.5, minor=True)
        ax3.set_xticks(pt.arange(n_colsC+1) - 0.5, minor=True)
        ax3.set_yticks(pt.arange(n_rowsC+1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax1.set_xlabel("{:d} columns".format(n_colsA), fontsize=LABEL_SIZE)
        ax1.set_ylabel("{:d} rows".format(n_rowsA), fontsize=LABEL_SIZE)
        ax2.set_xlabel("{:d} columns".format(n_colsB), fontsize=LABEL_SIZE)
        ax2.set_ylabel("{:d} rows".format(n_rowsB), fontsize=LABEL_SIZE)
        ax3.set_xlabel("{:d} columns".format(n_colsC), fontsize=LABEL_SIZE)
        ax3.set_ylabel("{:d} rows".format(n_rowsC), fontsize=LABEL_SIZE)


def plot_matrix_vector_multiplication_as_heatmap(A: pt.Tensor, B: pt.Tensor,
                                                 C: pt.Tensor,
                                                 annotate: bool = True):
    n_rowsA, n_colsA = A.size()
    n_rowsB, n_colsB = B.size()
    n_rowsC, n_colsC = C.size()
    fig = plt.figure(num=None, figsize=((n_colsA+n_colsB+n_colsC+1), 3),
                     dpi=Image_size, facecolor='w', edgecolor='k')
    spec = gridspec.GridSpec(
        ncols=3, nrows=1, width_ratios=[n_colsA, n_colsB, n_colsC],
        height_ratios=[3], figure=fig
    )
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax1.imshow(A)
    ax2.imshow(B)
    ax3.imshow(C)
    if annotate:
        for row_A in range(n_rowsA):
            for col_A in range(n_colsA):
                ax1.text(
                    col_A, row_A, "{:2.2f}".format(A[row_A, col_A]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
        for row_B in range(n_rowsB):
            for col_B in range(n_colsB):
                ax2.text(
                    col_B, row_B, "{:2.2f}".format(B[row_B, col_B]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
        for row_C in range(n_rowsC):
            for col_C in range(n_colsC):
                ax3.text(
                    col_C, row_C, "{:2.2f}".format(C[row_C, col_C]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
    for ax, in zip([ax1, ax2, ax3]):
        ax1.set_xticks(pt.arange(n_colsA+1) - 0.5, minor=True)
        ax1.set_yticks(pt.arange(n_rowsA+1) - 0.5, minor=True)
        ax2.set_xticks(pt.arange(n_colsB+1) - 0.5, minor=True)
        ax2.set_yticks(pt.arange(n_rowsB+1) - 0.5, minor=True)
        ax3.set_xticks(pt.arange(n_colsC+1) - 0.5, minor=True)
        ax3.set_yticks(pt.arange(n_rowsC+1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax1.set_xlabel("{:d} columns".format(n_colsA), fontsize=LABEL_SIZE)
        ax1.set_ylabel("{:d} rows".format(n_rowsA), fontsize=LABEL_SIZE)
        ax2.set_xlabel("{:d} columns".format(n_colsB), fontsize=LABEL_SIZE)
        ax2.set_ylabel("{:d} rows".format(n_rowsB), fontsize=LABEL_SIZE)
        ax3.set_xlabel("{:d} columns".format(n_colsC), fontsize=LABEL_SIZE)
        ax3.set_ylabel("{:d} rows".format(n_rowsC), fontsize=LABEL_SIZE)


def plot_matrix_transpose_as_heatmap(A: pt.Tensor, B: pt.Tensor,

                                     annotate: bool = True):
    n_rowsA, n_colsA = A.size()
    n_rowsB, n_colsB = B.size()
    fig = plt.figure(num=None, figsize=((n_colsA+n_colsB), 3),
                     dpi=Image_size, facecolor='w', edgecolor='k')
    spec = gridspec.GridSpec(
        ncols=2, nrows=1, width_ratios=[n_colsA, n_colsB],
        height_ratios=[3], figure=fig
    )
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax1.imshow(A)
    ax2.imshow(B)

    if annotate:
        for row_A in range(n_rowsA):
            for col_A in range(n_colsA):
                ax1.text(
                    col_A, row_A, "{:2.2f}".format(A[row_A, col_A]),
                    ha="center", va="center",
                    color="w", fontsize=ANNO_SIZE)
        for row_B in range(n_rowsB):
            for col_B in range(n_colsB):
                ax2.text(
                    col_B, row_B, "{:2.2f}".format(B[row_B, col_B]),
                    ha="center", va="center",
                    color="w", fontsize=ANNO_SIZE
                )
    for ax, in zip([ax1, ax2]):
        ax1.set_xticks(pt.arange(n_colsA+1) - 0.5, minor=True)
        ax1.set_yticks(pt.arange(n_rowsA+1) - 0.5, minor=True)
        ax2.set_xticks(pt.arange(n_colsB+1) - 0.5, minor=True)
        ax2.set_yticks(pt.arange(n_rowsB+1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax1.set_xlabel("{:d} columns".format(n_colsA), fontsize=LABEL_SIZE)
        ax1.set_ylabel("{:d} rows".format(n_rowsA), fontsize=LABEL_SIZE)
        ax2.set_xlabel("{:d} columns".format(n_colsB), fontsize=LABEL_SIZE)
        ax2.set_ylabel("{:d} rows".format(n_rowsB), fontsize=LABEL_SIZE)


def plot_SVD_as_heatmap(A: pt.Tensor, B: pt.Tensor,
                        C: pt.Tensor, annotate: bool = True):
    n_rowsA, n_colsA = A.size()
    n_rowsB, n_colsB = B.size()
    n_rowsC, n_colsC = C.size()
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(
        ncols=3, nrows=1, width_ratios=[n_colsA, n_colsB, n_colsC],
        height_ratios=[3], figure=fig
    )
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax1.imshow(A)
    ax2.imshow(B)
    ax3.imshow(C)
    if annotate:
        for row_A in range(n_rowsA):
            for col_A in range(n_colsA):
                ax1.text(
                    col_A, row_A, "{:2.2f}".format(A[row_A, col_A]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
        for row_B in range(n_rowsB):
            for col_B in range(n_colsB):
                ax2.text(
                    col_B, row_B, "{:2.2f}".format(B[row_B, col_B]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
        for row_C in range(n_rowsC):
            for col_C in range(n_colsC):
                ax3.text(
                    col_C, row_C, "{:2.2f}".format(C[row_C, col_C]),
                    ha="center", va="center", color="w", fontsize=ANNO_SIZE
                )
    for ax, in zip([ax1, ax2, ax3]):
        ax1.set_xticks(pt.arange(n_colsA+1) - 0.5, minor=True)
        ax1.set_yticks(pt.arange(n_rowsA+1) - 0.5, minor=True)
        ax2.set_xticks(pt.arange(n_colsB+1) - 0.5, minor=True)
        ax2.set_yticks(pt.arange(n_rowsB+1) - 0.5, minor=True)
        ax3.set_xticks(pt.arange(n_colsC+1) - 0.5, minor=True)
        ax3.set_yticks(pt.arange(n_rowsC+1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax1.set_xlabel("{:d} columns".format(n_colsA), fontsize=LABEL_SIZE)
        ax1.set_ylabel("{:d} rows".format(n_rowsA), fontsize=LABEL_SIZE)
        ax2.set_xlabel("{:d} columns".format(n_colsB), fontsize=LABEL_SIZE)
        ax2.set_ylabel("{:d} rows".format(n_rowsB), fontsize=LABEL_SIZE)
        ax3.set_xlabel("{:d} columns".format(n_colsC), fontsize=LABEL_SIZE)
        ax3.set_ylabel("{:d} rows".format(n_rowsC), fontsize=LABEL_SIZE)
