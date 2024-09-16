import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.auto_mpg import auto_mpg
from datasets.forest_fires import forestfires
from datasets.seoul_bike_sharing_demand import seoul_bike
from datasets.boston import boston
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from regression import linear_regression_model, cross_validation_scores

def feature_auto_mpg():
    dataset = auto_mpg.get_dataset()

    x = dataset.drop('mpg', axis=1)
    y = dataset['mpg']
    return x, y

def feature_forest_fires():
    data = forestfires.get_dataset()
    x = data.drop(columns=['area'])
    y = data['area']
    return x, y

def feature_seoul_bike():
    data = seoul_bike.get_dataset()
    x = data.drop('Rented Bike Count', axis=1)
    y = data['Rented Bike Count']
    return x, y

def feature_boston_housing():
    data = boston.get_dataset()
    x = data.drop('MEDV', axis=1)
    y = data['MEDV']
    return x, y
def run_auto_mpg():
    x, y = feature_auto_mpg()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)

def run_forest_fires():
    x, y = feature_forest_fires()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)

def run_seoul_bike():
    x, y = feature_seoul_bike()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)

def run_boston_housing():
    x, y = feature_boston_housing()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)

def main():
    print("Auto MPG Linear Regression Results:")
    run_auto_mpg()
    print("\nForest Fires Linear Regression Results:")
    run_forest_fires()
    print("\nSeoul Bike Dataset Linear Regression Results:")
    run_seoul_bike()
    print("\nBoston Housing Linear Regression Results:")
    run_boston_housing()

if __name__ == "__main__":
    main()
