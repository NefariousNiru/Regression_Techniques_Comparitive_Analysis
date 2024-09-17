Create a virtual environment
`python -m venv venv`
- On windows : `venv\Scripts\activate`
- On linux: `source venv/bin/activate`

Install package dependencies using
`pip install -r requirements.txt`

The structure of the project follows
- datasets (directory containing datasets and EDA)

Regression Models
    - lasso
    - linear
    - ridge
    - symbolic
    - transformed

- util (utility methods - eda, pre_processing, loading, metrics)

All regression models contain `run.py` which is the entry point for the application