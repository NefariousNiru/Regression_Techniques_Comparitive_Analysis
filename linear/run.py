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

def preprocess_auto_mpg():
    dataset = auto_mpg.get_dataset()
    columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
    data = pd.read_csv(dataset, names=columns, skipinitialspace=True, sep=" ", comment="\t")
    data = data.drop(columns="car_name")
    data = data.dropna()
    
    outliers = stats.zscore(data.select_dtypes(include=['int64','float64']))
    abs_outliers = np.abs(outliers)
    filter = (abs_outliers < 3).all(axis=1)
    data = data[filter]
    
    corr = data.corr()
    sns.heatmap(corr, cmap='Blues')
    plt.title("Correlation with Features")
    plt.show()
    
    x = data.drop('mpg', axis=1)
    y = data['mpg']
    return x, y

def preprocess_forest_fires():
    data = forestfires.get_dataset()
    data = data.drop_duplicates()
    
    outliers = stats.zscore(data.select_dtypes(include=['int64','float64']))
    abs_outliers = np.abs(outliers)
    filter = (abs_outliers < 3).all(axis=1)
    data = data[filter]
    
    data = pd.get_dummies(data, columns=['day','month'], drop_first=True)
    
    corr = data.corr()
    sns.heatmap(corr, cmap='Blues')
    plt.title("Correlation with Features")
    plt.show()
    
    x = data.drop(columns=['area'])
    y = data['area']
    return x, y

def preprocess_seoul_bike():
    data = seoul_bike.get_dataset()
    corr = data.corr()
    sns.heatmap(corr, cmap='Blues')
    plt.title("Correlation with Features")
    plt.show()
    
    x = data.drop('Rented Bike Count', axis=1)
    y = data['Rented Bike Count']
    return x, y
def preprocess_boston_housing():
    data = boston.get_dataset()
    outliers = stats.zscore(data.select_dtypes(include=['int64','float64']))
    abs_outliers = np.abs(outliers)
    filter = (abs_outliers < 3).all(axis=1)
    data = data[filter]
    
    corr = data.corr()
    sns.heatmap(corr, cmap='Blues')
    plt.title("Correlation with Features")
    plt.show()
    
    x = data.drop('MEDV', axis=1)
    y = data['MEDV']
    return x, y
def run_auto_mpg():
    x, y = preprocess_auto_mpg()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)

def run_forest_fires():
    x, y = preprocess_forest_fires()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)

def run_seoul_bike():
    x, y = preprocess_seoul_bike()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)
def run_boston_housing():
    x, y = preprocess_boston_housing()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr = linear_regression_model(x_train, x_test, y_train, y_test)
    cv_rmse = cross_validation_scores(lr, x, y)
    print("Cross-Validation RMSE:", cv_rmse)
def main():
    print("Auto MPG Dataset Linear Regression Results:")
    run_auto_mpg()
    print("\nForest Fires Dataset Linear Regression Results:")
    run_forest_fires()
    print("\nSeoul Bike Sharing Dataset Linear Regression Results:")
    run_seoul_bike()
    print("\nBoston Housing Dataset Linear Regression Results:")
    run_boston_housing()

if __name__ == "__main__":
    main()
