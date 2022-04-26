## rasopt - a HEC-RAS calibration tool
Rasopt is your one stop shop for efficient calibration of the HEC-RAS 2D flood model.
Rasopt is a Python wrapper that allows for calibration of model parameters using 
a suite of optimization algorithms, both out-of-the-box or user supplied. At it's core, rasopt 
creates an environment that facilitates easy running of the HEC-RAS model in a programmatic 
way, thus sidestepping the cumbersome GUI that is required to run HEC-RAS normally. 

### Installation
To install rasopt, run the following within a virtual environment of choice.

```
c:\> git clone https://github.com/alex-l-young/rasopt

c:\> cd rasopt

c:\rasopt> pip install .
```


### Required Libraries
rascontrol - rascontrol is not on PyPI and so must be installed separately using the following commands.

```
c:\> git clone https://github.com/mikebannis/rascontrol.git

c:\> cd rascontrol

c:\rascontrol> pip install .
```
