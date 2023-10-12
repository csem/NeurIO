# Installation
## Base requirements

NeurIO requires Python 3.6 or higher. It is recommended to use a virtual environment to install NeurIO.

## Installation using pip

This library is available as a preview, installation from pip is not yet available.

##  Installation from source

You can also install NeurIO from source, by cloning the repository and running the setup script:
```shell
git clone https://github.com/csem/NeurIO
cd NeurIO
python setup.py install
```

##  Dependencies

NeurIO has several dependencies, which are automatically installed when using pip. To install them manually, run the following command:
```shell
pip install -r requirements.txt
```

## NeurIO version

To check the version of NeurIO installed, access the `__version__` attribute of the package:
```python
import neurio
print(neurio.__version__)python
```

## Building the documentation

The NeurIO documentation is based on Sphinx and can be built using the following command:
```shell
cd docs
make html
```

Once built, the documentation can be placed in `neurio\docs\html`. Open `index.html` in a web browser to start using the documentation.