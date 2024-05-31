<!--
<p align="center">
  <img src="https://github.com/ziad-elmalki/ai4chem/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  ai4chem
</h1>

<p align="center">
    <a href="https://github.com/ziad-elmalki/ai4chem/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/ziad-elmalki/ai4chem/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/ai4chemistry">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/ai4chemistry" /></a>
    <a href="https://pypi.org/project/ai4chemistry">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/ai4chemistry" /></a>
    <a href="https://github.com/ziad-elmalki/ai4chem/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/ai4chemistry" /></a>
    <a href='https://ai4chemistry.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/ai4chemistry/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="https://codecov.io/gh/ziad-elmalki/ai4chem/branch/main">
        <img src="https://codecov.io/gh/ziad-elmalki/ai4chem/branch/main/graph/badge.svg" alt="Codecov status" /></a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /></a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' /></a>
    <a href="https://github.com/ziad-elmalki/ai4chem/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/></a>
</p>



## 💪 Getting Started

> This repo allows for the training of an XGBoost model, Fine-tunign Google's ViT and the training of PepNet, a convolutinal network, for the prediction of cell permeability of cylic peptides


### Command Line Interface

The ai4chemistry command line tool is automatically installed. It can
be used from the shell with the `--help` flag to show all subcommands:

```shell
ai4chemistry --help
```

> TODO show the most useful thing the CLI does! The CLI will have documentation auto-generated
> by `sphinx`.

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/ai4chemistry/) with:

```shell
pip install ai4chemistry
```
-->

The most recent code and data can be installed directly from GitHub with:

```shell
pip install git+https://github.com/ziad-elmalki/ai4chem.git
```

### To download the data:
On linux:
```shell
conda env create pepnet_linux.yml
```
On Ios:
```shell
conda env create pepnet_ios.yml
```

Then:
```shell
conda activate pepnet

python CycPept_Download.py
python CycPept_Cleaning.py
```
### To run the PepNet CNN:
```shell
python PepNet_DataGenerator.py
python PepNet.py
```

### To run the ridge reg:
```shell
conda env create -f map4.yml
conda activate map4
python Fingerprints.ipynb

conda activate pepnet
python Baseline_model.py
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/ziad-elmalki/ai4chem/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
git clone git+https://github.com/ziad-elmalki/ai4chem.git
cd ai4chem
pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
tox
```

Additionally, these tests are automatically re-run with each commit in a
[GitHub Action](https://github.com/ziad-elmalki/ai4chem/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
git clone git+https://github.com/ziad-elmalki/ai4chem.git
cd ai4chem
tox -e docs
open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

The documentation can be deployed to [ReadTheDocs](https://readthedocs.io) using 
[this guide](https://docs.readthedocs.io/en/stable/intro/import-guide.html).
The [`.readthedocs.yml`](.readthedocs.yml) YAML file contains all the configuration you'll need.
You can also set up continuous integration on GitHub to check not only that
Sphinx can build the documentation in an isolated environment (i.e., with ``tox -e docs-test``)
but also that [ReadTheDocs can build it too](https://docs.readthedocs.io/en/stable/pull-requests.html).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/ai4chemistry/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file
   configured to avoid the need for manual input at this step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.

</details>
