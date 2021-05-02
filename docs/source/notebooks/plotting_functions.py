import matplotlib.pyplot as plt

import torch as pt
import numpy as np

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120


pt.pi = pt.acos(pt.zeros(1)).item()*2  #PyTorch has no built in Pi function

t = pt.linspace(0, 6*pt.pi, 80)
Tm, Xm = pt.meshgrid(t, pt.linspace(-10, 10, 100))
dt = Tm[1,0]-Tm[0,0]


def plot_data(func, n):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    for i in range(12):
        ax.plot(Xm[0,:], pt.real(func[:, i]), c='C{}'.format(n-1), alpha=1-(i/11), label='$t_{{{temp}}}$'.format(temp=i))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel('$Re(f_{})$'.format(n))
    if n==4:
        ax.set_ylabel('Re(f1+f2+f3)')
    if n==5:
        ax.set_ylabel('Reconstucted data $\hat{X}$')
    ax.set_xlim(pt.min(Xm[0,:]), pt.max(Xm[0,:]))
    
def plot_singular_values(s_temp):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.set_ylabel('$s[i]$')
    ax.set_xlabel('$i$')
    plt.scatter(range(s_temp.shape[0]), s_temp, s=10, c='C3')
    plt.show()
    
def plot_eigenvalues(val):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    t = pt.linspace(0.0, pt.pi/2, 100)
    ax.plot(pt.cos(t), pt.sin(t), ls="--", color="k", lw=2, label='stable')
    growing = plt.fill_between(1.1*t, 1.1, color="k", alpha=0.05)
    decaying = plt.fill_between(pt.cos(t), pt.sin(t), color="k", alpha=0.2)

    plt.text(0.2, 0.2,'DMD mode is decaying')
    plt.text(0.55, 1,'DMD mode is growing')
    ax.annotate(' ', xy=(0.492,0.885), xytext=(0.35, 0.65), arrowprops=dict(facecolor='black', shrink=0.1, width=0.5, headwidth=4, headlength=6))
    plt.text(0.05, 0.65,'DMD mode is neither')
    plt.text(0.05, 0.6,'growing nor decaying')

    #plt.legend((decaying, stable, decaying), ('Decaying', 'Unchanging', 'Growing'))

    colors = ["C{:d}".format(i) for i in range(len(val))]
    ax.scatter(pt.real(val), pt.imag(val), color=colors, marker="o", s=100)
    for i in range(len(val)):
        ax.annotate(r"$\lambda_{:d}$".format(i+1), (pt.real(val[i])+0.02, pt.imag(val[i])+0.02))
    ax.set_xlabel(r"$Re(\lambda)$")
    ax.set_ylabel(r"$Im(\lambda)$")
    ax.set_xlim(0.0, 1.1)
    ax.set_ylim(0.0, 1.1)
    ax.set_aspect(1)
    
def plot_dmd_modes(x, modes):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for i in range(modes.shape[1]):
        ax.scatter(x, pt.real(modes[:, i]), s=8, label=r"${:s}_{:d}$".format(r"{\varphi}", i+1))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\varphi_i$")
    ax.set_xlim(pt.min(x), pt.max(x))
    ax.legend()
    
def plot_psi_in_time(psi, n):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(Tm[:,0], pt.real(psi[n-1,:]), c='C{}'.format(n-1), label='$Re \, \psi_{}$'.format(n))
    ax.plot(Tm[:,0], pt.imag(psi[n-1,:]), c='C{}'.format(n-1), label='$Im \, \psi_{}$'.format(n), alpha=0.3)
    ax.set_xlabel('t')
    ax.set_xlim(0, 20)
    ax.legend()
    fig.show()
    
def dmd(matrix, rank=None):
    U, s, Vt = pt.linalg.svd(matrix[:,:-1])
    U, s, V = U[:, :rank], s[:rank], Vt.conj().T[:,:rank]
    At = U.conj().T @ matrix[:, 1:] @ V @ pt.diag(1.0/s).type(pt.complex64)
    val, vec = np.linalg.eig(At)
    val, vec = pt.from_numpy(val), pt.from_numpy(vec)
    phi = matrix[:, 1:] @ V @ np.diag(1.0/s) @ vec
    return val, phi

def recon_data_from_DMD(f, r):
    '''DMD'''
    val4, phi4 = dmd(f,r)
    '''Time Dynamics'''
    b4 = (pt.pinverse(phi4) @ f[:,0]).T
    psi4 = pt.zeros(r, len(Tm[:,0]), dtype=pt.complex64)
    for i, t_temp in enumerate(t):    
        psi4[:,i] = pt.pow(val4, t_temp/dt) * b4
    '''Reconstruct Function'''
    g_hat = phi4 @ psi4
    return g_hat

def plot_recon_img(f, pics): #hard coded: I know this is bad practice
    fig, ax = plt.subplots(2, 3, figsize=(8, 4.5))
    if(pics>1):
        for i in range(3):
            im = ax[0][i].imshow(np.real(recon_data_from_DMD(f, r=(i+1)*2))) #data f, rank r
            ax[0][0].set_ylabel(r"$x$")
            ax[0][i].set_xticklabels([])
            ax[0][i].set_yticklabels([])
            ax[0][0].title.set_text("$rank=2$")
            ax[0][i].title.set_text("$={}$".format((i+1)*2))
            im = ax[1][i].imshow(np.real(recon_data_from_DMD(f, r=(i+1)*2+6))) #data f, rank r
            ax[1][i].set_xlabel(r"$t$")
            ax[1][0].set_ylabel(r"$x$")
            ax[1][i].set_xticklabels([])
            ax[1][i].set_yticklabels([])
            ax[1][i].title.set_text("$={}$".format((i+1)*2+6))
    else:
        im = ax.imshow(np.real(f)) #data f, rank r
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.title.set_text("$rank=1$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.get_yaxis().labelpad = 5
    cbar.ax.set_ylabel(r"$Re(\hat{f}_4(x))$", rotation=90)
    plt.show()
    
def plot_data_matrix_image(X, name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    im = ax.imshow(np.real(X))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.get_yaxis().labelpad = 5
    cbar.ax.set_ylabel(r"$Re({})$".format(name), rotation=90)
    plt.show()
    
def plot_sing_val_and_function(f, n):
    v, s, u = pt.linalg.svd(f)
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))
    ax[0].scatter(range(s.shape[0]), s, s=10, c='C3')
    ax[0].set_ylabel('$s[i]$')
    ax[0].set_xlabel('$i$')
    
    im = ax[1].imshow(np.real(f))
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$x$")
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.get_yaxis().labelpad = 5
    cbar.ax.set_ylabel(r"$Re(f_{}(x))$".format(n), rotation=90)
    plt.show()