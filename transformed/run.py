import pandas as pd
from sklearn.model_selection import train_test_split
import util
from datasets.auto_mpg import auto_mpg
from datasets.forest_fires import forestfires
from datasets.seoul_bike_sharing_demand import seoul_bike
from datasets.boston import boston
from transformed import regression
from extra import random_forest


def get_data(data, y):
    X, y = util.load.get_x_y(data, [y])
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

# Performs validations and runs the regression
def execute(data, y_label):
    results = {}

    X, y, X_train, X_test, y_train, y_test = get_data(data, y_label)

    regression.logarithmic(X, y, X_train, X_test, y_train, y_test, results)
    regression.square_root(X, y, X_train, X_test, y_train, y_test, results)
    regression.reciprocal(X, y, X_train, X_test, y_train, y_test, results)
    regression.box_cox(X, y, X_train, X_test, y_train, y_test, results)
    random_forest.random_forest(X, y, X_train, X_test, y_train, y_test, results)

    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)


def run_auto_mpg():
    auto_mpg_data = auto_mpg.get_dataset().drop(columns='brand')

    print("Performance including outliers")
    execute(auto_mpg_data.copy(), 'mpg')

    print("\nPerformance excluding outliers - Z Score")
    auto_mpg_data_no_outliers_z = util.pre_processing.drop_outliers_zscore(auto_mpg_data.copy())
    execute(auto_mpg_data_no_outliers_z, 'mpg')

    print("\nPerformance excluding outliers - Inter Quartile")
    auto_mpg_data_no_outliers_iq = util.pre_processing.drop_outliers_inter_quartile(auto_mpg_data.copy())
    execute(auto_mpg_data_no_outliers_iq, 'mpg')


def run_forest_fires():
    forest_fires_data = forestfires.get_dataset()

    print("Performance including outliers")
    execute(forest_fires_data.copy(), 'area')

    print("\nPerformance excluding outliers - Z Score")
    forest_fires_data_no_outliers_z = util.pre_processing.drop_outliers_zscore(forest_fires_data.copy())
    execute(forest_fires_data_no_outliers_z, 'area')

    print("\nPerformance excluding outliers - Inter Quartile")
    forest_fires_data_no_outliers_iq = util.pre_processing.drop_outliers_inter_quartile(forest_fires_data.copy())
    execute(forest_fires_data_no_outliers_iq, 'area')


def run_seoul_bike():
    seoul_bikes = seoul_bike.get_dataset()
    print("Performance including outliers")
    execute(seoul_bikes.copy(), 'Rented Bike Count')

    print("\nPerformance excluding outliers - Z Score")
    seoul_bikes_no_outliers_z = util.pre_processing.drop_outliers_zscore(seoul_bikes)
    execute(seoul_bikes_no_outliers_z, 'Rented Bike Count')

    print("\nPerformance excluding outliers - Inter Quartile")
    seoul_bikes_no_outliers_iq = util.pre_processing.drop_outliers_inter_quartile(seoul_bikes)
    execute(seoul_bikes_no_outliers_iq, 'Rented Bike Count')


def run_boston_housing():
    boston_data = boston.get_dataset()

    print("Performance including outliers")
    execute(boston_data.copy(), 'MEDV')

    print("\nPerformance excluding outliers - Z Score")
    boston_data_no_outliers_z = util.pre_processing.drop_outliers_zscore(boston_data.copy())
    execute(boston_data_no_outliers_z, 'MEDV')

    print("\nPerformance excluding outliers - Inter Quartile")
    forest_fires_data_no_outliers_iq = util.pre_processing.drop_outliers_inter_quartile(boston_data.copy())
    execute(forest_fires_data_no_outliers_iq, 'MEDV')


# Calls all the regression methods on each dataset
def run():
    run_auto_mpg()
    run_forest_fires()
    run_seoul_bike()
    run_boston_housing()

run()