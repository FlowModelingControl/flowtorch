# How to contribute to *flowTorch*

We appreciate all efforts contributing to the *flowTorch* project, may it be bug-fixes, feature contributions, feature suggestions, additional examples, or other kinds of improvements. If you would like to contribute, you may consider the following steps.

## 0. Open a new issue

It is always useful to open a new issue as a first step. The issue helps the developers to plan and organize developments and to provide quick feedback on potential problems or existing solutions. For example, it might be that the bug you are reporting has already been fixed on a development branch or that someone is already working on a similar feature to the one you are suggesting. *flowTorch* is still a rather small project, so there are typically few open issues. Nonetheless, you should give your issue a suitable label to follow common best practices (e.g., feature, bug, documentation, ...).

## 1. Fork the *flowTorch* repository and create a new branch

The typical workflow of forking and branching is very well described in the [GitHub documentation](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

## 2. Ensure code quality

*flowTorch* uses the [PyTorch library](https://pytorch.org/docs/stable/index.html) as backend for array-like data structures (tensors) and operations thereon. When implementing new features, try to rely as much as possible on the functionality offered by PyTorch instead of using NumPy, SciPy or similar libraries.

Most of the library contains [type hints](https://docs.python.org/3/library/typing.html). Type hints are not strictly necessary to run the code, but they make the lives of everybody much easier, so please use type hint in all parts of your code.

Python is a language that allows implementing operations with enormous complexity in a single line of code. Therefore, it is extremely important to provide a detailed documentation of new functionality containing all considerations the developer had in mind and also potential references or resources that were used as basis. *flowTorch* generates the documentation using [Sphinx](https://www.sphinx-doc.org/en/master/), and therefore, doc-stings should be formatted as [reStructuredText](https://docutils.sourceforge.io/rst.html).

We use PyLint to ensure proper code formatting. If VSCode is your editor of choice, have a look at the [documentation for linting](https://code.visualstudio.com/docs/python/linting) to setup automated code formatting.

## 3. Provide unit tests

If new features are added, accompanying unit tests should be provided. We use [PyTest](https://docs.pytest.org/en/6.2.x/) for testing. If the tests require additional datasets, please make sure that you have the permission to share the data such that the new data can be added to the *flowTorch* datasets in the next release. It might be necessary in some cases to create fake data (data that behave the same way as real data but that might be smaller and not protected).

## 4. Push changes and create a pull-request

To complete your contribution, create a new [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) for your feature or bug-fix. The changes will then be tested by someone with write access to the main repository before they are merged. If datasets are required, please provide a download link such that unit tests or examples can be executed.

**Thank you for considering to contribute to the *flowTorch* project!**