---
title: 'flowTorch - a Python library for analysis and reduced-order modeling of fluid flows'
tags:
  - Python
  - PyTorch
  - fluid flows
  - reduced-order modeling
  - modal analysis
authors:
  - name: Andre Weiner
    orcid: 0000-0001-5617-1560
    affiliation: 1
  - name: Richard Semaan
    orcid: 0000-0002-3219-0545
    affiliation: 1
affiliations:
 - name: Technische Universit\"at Braunschweig, Institute of Fluid Mechanics, Flow Modeling and Control Group
   index: 1
date: 05 August 2021
bibliography: paper.bib
---

# Summary

The `flowTorch` library enables researchers to access, analyze, and model
fluid flow data from experiments or numerical simulations. Instead of a black-box end-to-end solution,
`flowTorch` provides modular components allowing to assemble transparent and reproducible workflows with ease. Popular
data formats for fluid flows like [OpenFOAM](https://www.openfoam.com/), [VTK](https://vtk.org/), or
[DaVis](https://www.lavision.de/en/products/davis-software/) may be accessed via a common interface in
a few lines of Python code. Internally, the data are organized as [PyTorch](https://pytorch.org/) tensors.
Relying on PyTorch tensors as primary data structure enables fast array operations, parallel processing on
CPU and GPU, and exploration of novel deep learning-based analysis and modeling approaches.
The `flowTorch` packages also includes a collection of Jupyter notebooks demonstrating  how to apply the
library components in a variety of different use cases, e.g., finding coherent flow structures with modal analysis or creating
reduced-order models.

# Statement of need

Thanks to the increased processing power of modern hardware, fluid flow experiments as well as numerical simulations
are producing vast amounts of highly resolved, complex data. Those data offer great opportunities to improve complex technical application or to understand natural phenomena. However, gaining insights from the data is becoming increasingly challenging. As data sizes grow, more attention has to be paid to the
data format. While most researchers prefer simple text-encoded comma-separated value (CSV) files, big datasets require
special binary formats, such as [HDF5](https://www.hdfgroup.org/solutions/hdf5/) or [NetCDF](https://en.wikipedia.org/wiki/NetCDF).
If the data are associated with a structured or an unstructured mesh, VTK files are a popular choice. Other simulation libraries for
fluid flows, like OpenFOAM, organize mesh and field data in custom folder and file structures. CSV, VTK, or OpenFOAM data may come
as binary or text-encoded files. This list is by no means comprehensive in terms of available formats and presents only the tip of the iceberg.

A common research task may be to compare and combine different data sources of the same fluid flow problem for cross-validation
or to leverage each source's strengths in different kinds of analysis. A typical example would be to compare or combine particle image velocity (PIV) data with sampled
planes extracted from a numerical simulation. The simulation offers greater details and additional field information, while the PIV experiment is more trustworthy since it
is closer to the real application. The PIV data may have to be processed and cleaned before using it in consecutive analysis steps. When confronted with such ideas, researchers will spend a significant amount of time,
if not most of the time, accessing, converting, and processing the data with different tools and different formats to finally analyze the data in yet another tool.
Text-encoded file format might be convenient at first when exchanging data between tools, but for large datasets the additional conversion is unsuitable.

`flowTorch` aims to simplify access to data by providing a unified interface to various data formats via the subpackage `flowtorch.data`. Accessing data from a
distributed OpenFOAM simulation is as easy as loading VTK or PIV data and requires only a few lines of Python code. All field data
are converted internally to PyTorch tensors [@paszke2015]. Once the data are available as PyTorch tensors, further processing steps like scaling, clipping, masking, splitting, or merging are readily available as single function calls. The same is true for computing statistics, performing modal analysis,
or building reduced-order models via the subpackages `flowtorch.analysis` and `flowtorch.rom`. Computationally intensive tasks may be offloaded to the GPU if needed, which greatly accelerates parameter studies. The entire analysis workflow described in the previous section can be performed in a single ecosystem sketched in \autoref{fig:ft_structure}. Moreover, re-using an analysis pipeline in a different problem setting is straightforward.

![Components of flowTorch and library dependencies.\label{fig:ft_structure}](media/flowtorch_components_plain.pdf){ width=90% }

Another more general issue we want to address is the reproducibility of research outcomes. Popular algorithms like proper orthogonal
decomposition (POD)[@brunton2019], or dynamic mode decomposition (DMD)[@schmid2010; @kutz2016], may be relatively easy to
implement with libraries like NumPy, SciPy, or PyTorch. However,
applying these algorithms to real datasets typically requires several pre-processing steps, like cropping, clipping, or normalizing the
data, and careful tuning of the algorithms' free parameters (hyperparameters). Therefore, it is often unclear which exact steps were
taken to produce the reported results and how robust the results are to changes in the free parameters or the data. Even if the authors are willing to provide
more details, essential information may not be accessible due to black-box (closed-source) analysis tools used somewhere in the process.

With `flowTorch`, we attempt to make analysis and modeling workflows accessible, streamlined, and transparent in several ways:

- we provide Jupyter labs with start-to-end workflows, including short explanations for each step taken in the process; the notebooks' content varies from
toy examples through common benchmark problems to the analysis of real turbulent flow data; the datasets used in the notebooks are also part of
the library
- the library is modular and often wraps around other libraries to make them easier to use; a few lines of Python code are sufficient to implement
a basic workflow; the modular structure and the rich documentation of the source code simplify writing extensions and enable quick automated experimentation

Ultimately, our goal is to reduce redundant work as much as possible and enable users to focus on what matters - understanding and modeling flow dynamics.

# Examples

In this section, we demonstrate two applications of `flowTorch`. In the first example, DMD is employed to identify relevant modes in a transonic flow displaying shock-boundary-layer interactions. In the second example, a reduced-order model (ROM) of the flow past a circular cylinder is constructed employing cluster-based network modeling (CNM) [@fernex2021]. Both examples are also available as Jupyter labs and in the `flowTorch` documentation.

## DMD analysis of airfoil surface data

For this example, we need only a handful of `flowTorch` components.
```
import torch as pt
from flowtorch import DATASETS
from flowtorch.data import CSVDataloader, mask_box
from flowtorch.analysis import DMD
```
`DATASETS` is a dictionary holding names and paths of all available datasets. The `CSVDataloader` provides easy access to the data, and the `mask_box` function allows selecting only a spatial subset of the raw data. As the name suggests, the `DMD` class enables us to perform a DMD analysis.

The dataset we use here consists of surface pressure coefficient distributions sampled over a NACA-0012 airfoil in transonic flow conditions. At a Reynolds number of $Re=10^6$, a Mach number of $Ma=0.75$ and $\alpha = 4^\circ$ angle of attack, the flow displays a so-called shock buffet on the upper side of the airfoil. The shock buffet is a self-sustained unsteady interaction between the shock and the boundary layer separation. Our aim is to extract flow structures (modes) associated with the buffet phenomenon.

A code snippet to read the data, mask part of it, and build the data matrix reads:
```
...
path = DATASETS["csv_naca0012_alpha4_surface"]
loader = CSVDataloader.from_foam_surface(
    path, "total(p)_coeff_airfoil.raw", "cp")
vertices = loader.vertices
vertices /= (vertices[:, 0].max() - vertices[:, 0].min())
mask = mask_box(vertices,
    lower=[-1.0, 0.0, -1.0], upper=[0.9999, 1.0, 1.0])
points_upper = mask.sum().item()
data_matrix = pt.zeros((points_upper, len(times)), dtype=pt.float32)
for i, time in enumerate(times):
    snapshot = loader.load_snapshot("cp", time)
    data_matrix[:, i] = pt.masked_select(snapshot, mask)
```
The `CSVDataloader` has a class method designed to read raw sample data created by OpenFOAM simulations. Every `Dataloader` implementation provides access to one or multiple snapshots of one or multiple fields and the associated vertices. The airfoil coordinates are typically normalized with the chord length, which is the difference between largest and smallest value of the $x$-coordinate in the present example. The data contain pressure coefficients from both upper and lower surfaces, so we create a spatial mask to extract values from the upper surface. The DMD expects a data matrix as input whose columns are individual snapshots. Therefore, we allocate a new 2D tensor with as many rows as selected points and as many columns as selected snapshots (the data loader also provides access to the available write time - not shown here). Finally, we loop over the snapshot times and fill the data matrix.

Creating a new `DMD` instance automatically performs the mode decomposition based on the provided input. We can analyze
the obtained spectrum and the associated modes. The modes have real and imaginary parts, which are equally important for the
reconstruction of the flow field. It is usually enough to visualize either real or imaginary part for the physical interpretation of modes.
```
dmd = DMD(data_matrix, dt, rank=200)
amplitudes = dmd.amplitudes
frequencies = dmd.frequency
modes_real = dmd.modes.real
```
In contrast to POD, the DMD modes are not sorted by their variance, but rather form a spectrum.
\autoref{fig:dmd} presents the real part of three spatial modes with the largest amplitudes. Also shown is their corresponding frequency.

![Real part of three dominant DMD modes over the upper surface of a NACA-0012 airfoil. The shock is located at $x/c\approx 0.25$.\label{fig:dmd}](media/dmd_modes_airfoil_cp.pdf){ width=90% }

## CNM of the flow past a circular cylinder

This example demonstrates how to model a flow using the CNM algorithm [@fernex2021]. In `flowTorch`, creating a ROM always consists of three step: i) encoding/reduction, ii) time evolution, and iii) decoding/reconstruction. In the code snippet below, we use an encoder based on the singular value decomposition (SVD) to reduce the dimensionality of the original snapshot sequence, and then predict the temporal evolution and reconstruct the flow over the period of $1s$.

```
...
from flowtorch.rom import CNM, SVDEncoder
# load data
...
encoder = SVDEncoder(rank=20)
info = encoder.train(data_matrix)
reduced_state = encoder.encode(data_matrix)
cnm = CNM(reduced_state, encoder, dt, n_clusters=20, model_order=4)
prediction = cnm.predict(data_matrix[:, :5], end_time=1.0, step_size=dt)
```
The `predict` function computes the temporal evolution in the reduced state space and automatically performs the reconstruction. If we are only interested in the phase space, we can use `predict_reduced` instead, and reconstruct selected states using the encoder's `decode` method. The temporal evolution in the phase-space is displayed in \autoref{fig:cnm}.

![Phase-space representation of data clustering (large dots) and trajectory; the smaller dots mark interpolated time steps and are colored by their cluster affiliation.\label{fig:cnm}](media/cnm_cluster_transition.pdf){ width=70% }

# Acknowledgements

The authors gratefully acknowledge financial support by the German Research Foundation (DFG) received within the research unit [FOR 2895](https://www.for2895.uni-stuttgart.de/en/) *Unsteady flow and interaction phenomena at high speed stall conditions*.

# References