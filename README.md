![FOR2895Logo](media/for2895_logo.png)

# flowTorch

**flowTorch** - a Python library for analysis and reduced order modeling of fluid flows

*flowTorch* is developed primarily by [@AndreWeiner](https://github.com/AndreWeiner) in the [Flow Modeling and Control group](https://www.tu-braunschweig.de/en/ism/research-workgroups/flow-modelling-and-control) led by [Richard Semaan](https://www.tu-braunschweig.de/en/ism/research/flow-modelling-and-control/staff/semaan). The development is financed by the German Research Foundation (DFG) within the research program [FOR 2895](https://www.for2895.uni-stuttgart.de/)


> unsteady flow and interaction phenomena at high speed stall conditions

with the primary goal to investigate flow conditions that lead to [buffeting](https://en.wikipedia.org/wiki/Aeroelasticity#Buffeting) at airfoils in the transonic flow regime. The animation below shows the shock buffet on a NACA-0012 airfoil at *Re=10^7*, *Ma=0.75*, and 4 degrees angle of attack. The simulation was conducted in OpenFOAM; follow [this link](https://github.com/AndreWeiner/naca0012_shock_buffet) for more information about the setup.

https://user-images.githubusercontent.com/8482575/120886182-f2b78800-c5ec-11eb-9b93-efb9a139c431.mp4

## Why *flowTorch*?

The *flowTorch* project was started to make the analysis and modeling of fluid data **easy** and **accessible** to everyone. The library design intends to strike a balance between **usability** and **flexibility**. Instead of a monolithic, black-box analysis tool, the library offers modular components that allow assembling custom analysis and modeling workflows with ease. For example, performing a dynamic mode decomposition (DMD) of a transient *OpenFOAM* simulation looks as follows:

```
import torch as pt
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis.dmd import DMD

path = DATASETS["of_cylinder2D_binary"]
loader = FOAMDataloader(path)

# select a subset of the available snapshots
times = loader.write_times
window_times = [time for time in times if float(time) >= 4.0]

# load vertices, discard z-coordinate, and create a mask
vertices = loader.vertices[:, :2]
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
|flowtorch.data | data loading, domain reduction (masked selection) |
| flowtorch.analysis | algorithms for dimensionality reduction, including *proper orthogonal decomposition* (POD), *dynamic mode decomposition* (DMD), autoencoders, and variants thereof |
| flowtorch.rom | reduced-order modeling using [cluster-based network models (CNM)](https://github.com/fernexda/cnm) |

*flowTorch* uses the [PyTorch](https://github.com/pytorch/pytorch) library as a backend for data structures, data types, and linear algebra operations on CPU and GPU. Some cool features of *flowTorch* include:

- data accessors return PyTorch tensors, which can be used directly within your favorite machine learning library, e.g., *PyTorch*, *SkLearn* or *Tensorflow*
- most algorithms run on CPU as well as on GPU
- mixed-precision operations (single/double); switching to single precision makes your life significantly easier when dealing with large datasets
- user-friendly Python library that integrates easily with popular tools and libraries like *Jupyterlab*, *Matplotlib*, *Pandas*, or *Numpy*
- a rich tutorial collection to help you getting started
- interfaces to common data formats like [OpenFOAM](https://www.openfoam.com/), [VTK](https://vtk.org/) (for Flexi and SU2), [TAU](https://www.dlr.de/as/desktopdefault.aspx/tabid-395/526_read-694/), [iPSP](https://www.dlr.de/as/en/desktopdefault.aspx/tabid-183/251_read-13334/), CSV (for DaVis PIV data and raw OpenFOAM output)

*flowTorch* can be also used easily in combination with existing Python packages for analysis and reduced-order modeling thanks to the interoperability between PyTorch and NumPy. Great examples are (by no means a comprehensive list):

- [PyDMD](https://github.com/mathLab/PyDMD) - Python Dynamic Mode Decomposition
- [PySINDy](https://github.com/dynamicslab/pysindy) - sparse identification of nonlinear dynamical systems from data

## Getting started

The easiest way to install *flowTorch* is as follows:
```
# install via pip
pip3 install git+https://github.com/FlowModelingControl/flowtorch
# to uninstall flowTorch, run
pip3 uninstall flowtorch
```
Alternatively, you can also clone the repository manually by running
```
git clone git@github.com:FlowModelingControl/flowtorch.git
```
and install the dependencies listed in *requirements.txt*:
```
pip3 install -r requirements.txt
```

To get an overview of what *flowTorch* can do for you, have a look at the [online documentation](https://flowmodelingcontrol.github.io/flowtorch-docs/1.0/index.html). The examples presented in the online documentation are also contained in this repository. In fact, the documentation is a static version of several [Jupyter labs](https://jupyter.org/) with start-to-end analyses. If you are interested in an interactive version of one particular example, navigate to `./docs/source/notebooks` and run `jupyter lab`. Note that to execute some of the notebooks, the **corresponding datasets are required**. The datasets can be downloaded [here](https://cloudstorage.tu-braunschweig.de/getlink/fiYBqV7Qq1cAxLBsxQpPBvsw/datasets_15_10_2021.tar.gz) (~2.5GB). If the data are only required for unit testing, a reduced dataset may be downloaded [here](https://cloudstorage.tu-braunschweig.de/getlink/fiRa7B4bNX8EcWuybbS3fXsL/datasets_minimal_15_10_2021.tar.gz) (~341MB). Download the data into a directory of your choice and navigate into that directory. To extract the archive, run:
```
# full dataset
tar xzf datasets_15_10_2021.tar.gz
# reduced dataset
tar xzf datasets_minimal_15_10_2021.tar.gz
```
To tell *flowTorch* where the datasets are located, define the `FLOWTORCH_DATASETS` environment variable:
```
# add export statement to bashrc; assumes that the extracted 'datasets' or 'datasets_minimal'
# folder is located in the current directory
# full dataset
echo "export FLOWTORCH_DATASETS=\"$(pwd)/datasets/\"" >> ~/.bashrc
# reduced dataset
echo "export FLOWTORCH_DATASETS=\"$(pwd)/datasets_minimal/\"" >> ~/.bashrc
# reload bashrc
. ~/.bashrc
```

## Development
### Documentation

To build the flowTorch documentation, the following additional packages are required:
```
pip3 install sphinx sphinx_rtd_theme nbsphinx recommonmark
```
To build the HTML version of the API documentation, navigate to `./docs` and run:
```
make html
```

### Unit testing
All sub-packages contain unit tests, which require the installation of PyTest:
```
pip3 install pytest
```
Moreover, the flowTorch datasets must be downloaded and referenced as described in the previous section.
To run all unit tests of all sub-packages, execute:
```
pytest flowtorch
```
You can also execute all tests in a sub-package, e.g., data
```
pytest flowtorch/data
```
or run individual test modules, e.g.,
```
pytest flowtorch/data/test_FOAMDataloader.py
```

## Getting help

If you encounter any issues using *flowTorch* or if you have any questions regarding current and future development plans, please use the repository's [issue tracker](https://github.com/FlowModelingControl/flowtorch/issues). Consider the following steps before and when opening a new issue:

0. Have you searched for similar issues that may have been already reported? The issue tracker has a *filter* function to search for keywords in open issues.
1. Click on the green *New issue* button in the upper right corner and describe your problem as detailed as possible. The issue should state what **the problem** is, what the **expected behavior** should be, and, maybe, suggest a **solution**. Note that you can also attach files or images to the issue.
2. Select a suitable label from the drop-down menu called *Labels*.
3. Click on the green *Submit new issue* button and wait for a reply.

## Reference

If *flowTorch* aids your work, you may support our work by referencing this repository.

## License

*flowTorch* is [GPLv3](https://en.wikipedia.org/wiki/GNU_General_Public_License)-licensed; refer to the [LICENSE](https://github.com/FlowModelingControl/flowtorch/blob/main/LICENSE) file for more information.

