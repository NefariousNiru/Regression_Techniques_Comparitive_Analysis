# This project aims to model 4 datasets
- Boston Housing
- Auto MPG
- Seoul Bike Share Demand
- Forest Fires

## The objective is to test the model on various regression techniques.
- Linear
- Lasso
- Ridge
- Transform
    - Box Cox
    - Logarithmic
    - Reciprocal
    - Square Root
- Symbolic 

# Steps to run
Install Pycharm

Create a virtual environment using Pycharm UI instructions. 

Install package dependencies using
`pip install -r requirements.txt`

# The structure of the project follows
- datasets (directory containing datasets and EDA)

Regression Models
    - lasso
    - linear
    - ridge
    - symbolic
    - transformed
    - feature_selection
    - random forest (extra)

- util (utility methods - eda, pre_processing, loading, metrics)

All regression models contain `run.py` which is the entry point for the application, except symbolic regression which can be run using `regression.py`

# All dependencies are listed `requirements.txt` - Install package dependencies using
`pip install -r requirements.txt`