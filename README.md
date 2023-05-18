## rasopt - a HEC-RAS calibration tool
Rasopt is your one stop shop for efficient calibration of the HEC-RAS 2D flood model.
Rasopt is a Python wrapper that allows for calibration of model parameters using 
a suite of optimization algorithms, both out-of-the-box or user supplied. At it's core, rasopt 
creates an environment that facilitates easy running of the HEC-RAS model in a programmatic 
way, thus sidestepping the cumbersome GUI that is required to run HEC-RAS normally. 

### Installation
To install rasopt, run the following within a virtual environment of choice.

```
C:\> git clone https://github.com/alex-l-young/rasopt

C:\> cd rasopt

C:\rasopt> pip install .
```

### Python Version
Rasopt has been tested with Python version 3.9. There may be some flexibility with versioning, however, the GDAL and
Rasterio wheel files are for v3.9, so new wheel files will need to be downloaded for different Python versions.
GDAL and Rasterio wheel files for different Python version can be found 
[here](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

### Required Libraries

**rascontrol** - rascontrol is not on PyPI and so must be installed separately using the following commands.

```
C:\> git clone https://github.com/mikebannis/rascontrol.git

C:\> cd rascontrol

C:\rascontrol> pip install .
```

**GDAL** - GDAL must be installed from the wheel file included in the whls directory.

In the virtual environment you plan to use for rasopt, perform the following:
```
(rasopt-venv) C:\rasopt> pip install whls/GDAL-3.4.2-cp39-cp39-win_amd64.whl
```

**Rasterio** - Rasterio must be installed from the wheel file included in the whls directory after installing GDAL.

In the virtual environment you plan to use for rasopt, perform the following:
```
(rasopt-venv) C:\rasopt> pip install whls/rasterio-1.2.10-cp39-cp39-win_amd64.whl
```

