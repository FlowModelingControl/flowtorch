![FOR2895Logo](media/for2895_logo.png)

# flowTorch

**flowTorch** - a Python library for analysis and reduced order modeling of fluid flows

*flowTorch* is developed primarily by the [Flow Modeling and Control group](https://www.tu-braunschweig.de/en/ism/research-workgroups/flow-modelling-and-control) led by [Richard Semaan](https://www.tu-braunschweig.de/en/ism/research/flow-modelling-and-control/staff/semaan). The development is financed by the German Research Foundation (DFG) within the research program [FOR 2895](https://www.for2895.uni-stuttgart.de/)


> unsteady flow and interaction phenomena at high speed stall conditions

with the primary goal to investigate flow conditions that lead to [buffeting](https://en.wikipedia.org/wiki/Aeroelasticity#Buffeting) at airfoils in the transonic flow regime. The animation below shows the shock buffet on a NACA-0012 airfoil at *Re=10^7*, *Ma=0.75*, and 4 degrees angle of attack. The simulation was conducted in OpenFOAM; follow [this link](https://github.com/AndreWeiner/naca0012_shock_buffet) for more information about the setup.

https://user-images.githubusercontent.com/8482575/120886182-f2b78800-c5ec-11eb-9b93-efb9a139c431.mp4

## Why *flowTorch*?

The *flowTorch* project was started with the intention to strike a balance between **usability** and **flexibility**. Instead of a monolithic, black-box analysis tool, the library offers modular components that allow assembling custom analysis and modeling workflows with ease. For example, performing a dynamic mode decomposition (DMD) of a transient *OpenFOAM* simulation looks as follows:

```
import torch as pt
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis.dmd import DMD

loader = FOAMDataloader("run/flow_past_cylinder/")

# select a subset of the available snapshots
times = loader.write_times()
window_times = [time for time in times if float(time) >= 4.0]

# load vertices, discard z-coordinate, and create a mask
vertices = loader.get_vertices()[:, :2]
mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])

# assemble the data matrix
data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
for i, time in enumerate(window_times):
    # load the vorticity vector field, take the z-component [:, 2], and apply the mask
    data_matrix[:, i] = pt.masked_select(loader.load_snapshot("vorticity", time)[:, 2], mask)

# perform DMD
dmd = DMD(data_matrix, rank=19)
# analyse dmd.modes or dmd.eigvals
# ...
```

Currently, the following sub-packages are under active development. Note that some of the components are not yet available in the public release because further developments and testing are required:

| package | content |
| :------ | :-------|
|flowtorch.data | data loading, domain reduction (masked selection), and sampling, e.g., using *sparse spatial sampling* (S<sup>3</sup>) |
| flowtorch.analysis | algorithms for dimensionality reduction, including *proper orthogonal decomposition* (POD), *dynamic mode decomposition* (DMD), autoencoders, and variants thereof |
| flowtorch.rom | reduced-order modeling using [cluster-based network models (CNM)](https://github.com/fernexda/cnm); to be added soon |

*flowTorch* uses the [PyTorch](https://github.com/pytorch/pytorch) library as a backend for data structures, data types, and linear algebra operations on CPU and GPU. Some features in which *flowTorch* differs from similar packages include:

- parallel data processing on multiple CPUs and GPUs
- mixed-precision operations (single/double)
- user-friendly Python library that integrates easily with popular tools and libraries like *Jupyterlab*, *Matplotlib*, *Pandas*, or *Numpy*
- a tutorial collection to help you getting started
- interfaces to common data formats like [OpenFOAM](https://www.openfoam.com/), [CGNS](https://cgns.github.io/) (more are on the way)

## Getting started

The easiest way to install *flowTorch* is as follows:
```
# clone the repository
git clone git@github.com:AndreWeiner/flowtorch.git
# build the wheel
python3 setup.py bdist_wheel
# install flowTorch with pip
pip3 install dist/flowTorch-0.1-py3-none-any.whl
# to uninstall flowTorch, run
pip3 uninstall flowtorch
```

The repository contains a collection of examples as part of the documentation. To open the [Jupyter labs](https://jupyter.org/), navigate to `./docs/source/notebooks` and run `jupyter lab`. Note that to execute some of the notebooks, the corresponding datasets are required, which are not included in this repository. Feel free to get in touch if you are interested in the datasets.

## Development

For documentation and testing, the following additional packages are required:
```
pip3 install sphinx sphinx_rtd_theme nbsphinx pytest recommonmark
```
The build the HTML version of the API documentation, navigate to `./docs` and run:
```
make html
```
To perform unit testing, execute `pytest` in the repository's top-level folder. Note that additional test data are requirement for many of the tests. There are scripts located at `test/test_data` to create the test data yourself. Otherwise, feel free to get in touch.

## Getting help

If you encounter any issues using *flowTorch* or if you have any questions regarding current and future development plans, please use the repository's [issue tracker](https://github.com/AndreWeiner/flowtorch/issues). Consider the following steps before and when opening a new issue:

0. Have you searched for similar issues that may have been already reported? The issue tracker has a *filter* function to search for keywords in open issues.
1. Click on the green *New issue* button in the upper right corner and describe your problem as detailed as possible. The issue should state what **the problem** is, what the **expected behavior** should be, and, maybe, suggest a **solution**. Note that you can also attach files or images to the issue.
2. Select a suitable label from the drop-down menu called *Labels*.
3. Click on the green *Submit new issue* button and wait for a reply.

## Reference

**TODO:** add citeable reference

## License

*flowTorch* is [GPLv3](https://en.wikipedia.org/wiki/GNU_General_Public_License)-licensed; refer to the [LICENSE](https://github.com/AndreWeiner/flowtorch/blob/main/LICENSE) file for more information.

