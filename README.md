# xrom-os
xROM-OS - a Python library for flow analysis and reduced order modeling
create table of content

## Overview

purpose of xROM
optimized for large simulation data
point to documentation


## Installation

### Singularity or Docker (preferred)

### Local installation

PyPi packages for documentation:
```
pip install sphinx sphinx_rtd_theme nbsphinx
```
## Getting started

quick example, reference to documentation, reference to notebooks

## Getting help

search in issues
open new issue, impose structure

## Creating Singularity and Docker images

## Design ideas

- create masking class to perform PCA on a reduced domain
- create sub-mesh class that samples original data to new mesh
- S3 a third approach to reduce data and computational costs
- maybe: Datapipeline class that chains dataloader, masking, sub-mesh, S3, ...

- POD class uses dedicated writer class to write spatial modes?
- POD class can use pre-computed decomposition?
- POD.compute_modes(mode="snapshot/direct/data", solver="svd/eig/random")

## Contributors


