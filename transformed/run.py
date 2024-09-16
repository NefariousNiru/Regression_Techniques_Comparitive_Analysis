import pandas as pd
from sklearn.model_selection import train_test_split
import util
from datasets.auto_mpg import auto_mpg
from datasets.forest_fires import forestfires
from datasets.seoul_bike_sharing_demand import seoul_bike
from datasets.boston import boston
from transformed import regression

def get_data(data, y, scale, in_sample, cross_val, x_y_split):
    if scale:
        # Scale numeric data
        data = util.pre_processing.scale(data)

    # Get X, Y
    X, y = util.load.get_x_y(data, [y])

    # Validations
    if x_y_split:
        return train_test_split(X, y, test_size=0.2, random_state=42)
    if cross_val:
        return NotImplemented
    if in_sample:
        return X,y


def execute(data, y_label):
    results = {}

    X_train, X_test, y_train, y_test = get_data(data, y_label, False, False, False, True)
    regression.logarithmic(X_train, X_test, y_train, y_test, results)

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = get_data(data, y_label, True, False, False,True)
    regression.linear(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, results)
    regression.polynomial(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, results, 2)

    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None):
        print(metrics_df)


def run_auto_mpg():
    auto_mpg_data = auto_mpg.get_dataset().drop(columns='brand')

    print("Performance including outliers")
    execute(auto_mpg_data.copy(), 'mpg')

    print("\nPerformance excluding outliers")
    auto_mpg_data_no_outliers = util.pre_processing.drop_outliers(auto_mpg_data)
    execute(auto_mpg_data_no_outliers, 'mpg')


def run_forest_fires():
    forest_fires_data = forestfires.get_dataset()

    print("Performance including outliers")
    execute(forest_fires_data.copy(), 'area')

    print("\nPerformance excluding outliers")
    forest_fires_data_no_outliers = util.pre_processing.drop_outliers(forest_fires_data)
    execute(forest_fires_data_no_outliers, 'area')

def run_seoul_bike():
    seoul_bikes = seoul_bike.get_dataset()
    print("Performance including outliers")
    execute(seoul_bikes.copy(), 'Rented Bike Count')

    print("\nPerformance excluding outliers")
    seoul_bikes_no_outliers = util.pre_processing.drop_outliers(seoul_bikes)
    execute(seoul_bikes_no_outliers, 'Rented Bike Count')

def run_boston_housing():
    boston_data = boston.get_dataset()
    print("Performance including outliers")
    execute(boston_data.copy(), 'MEDV')

    print("\nPerformance excluding outliers")
    boston_data_no_outliers = util.pre_processing.drop_outliers(boston_data)
    execute(boston_data_no_outliers, 'MEDV')

def run():
    # run_auto_mpg()
    # run_forest_fires()
    # run_seoul_bike()
    run_boston_housing()

run()