import pandas as pd
from sklearn.model_selection import train_test_split
from feature_selection.regression import forward_selection, backward_selection, stepwise_selection
import util
from datasets.auto_mpg import auto_mpg
from datasets.forest_fires import forestfires
from datasets.seoul_bike_sharing_demand import seoul_bike
from datasets.boston import boston


def get_data(data, y):
    X, y = util.load.get_x_y(data, [y])
    return X, y


def execute(data, y_label):
    results = {}

    X, y = get_data(data, y_label)

    # Forward Selection
    selected_features_forward, r_squared_forward, adj_r_squared_forward = forward_selection(X, y)
    results['Forward Selection'] = {
        'Selected Features': selected_features_forward,
        'R Squared': r_squared_forward,
        'Adjusted R Squared': adj_r_squared_forward
    }

    # Backward Selection
    selected_features_backward, r_squared_backward, adj_r_squared_backward = backward_selection(X, y)
    results['Backward Selection'] = {
        'Selected Features': selected_features_backward,
        'R Squared': r_squared_backward,
        'Adjusted R Squared': adj_r_squared_backward
    }

    # Stepwise Selection
    selected_features_stepwise, r_squared_stepwise, adj_r_squared_stepwise = stepwise_selection(X, y)
    results['Stepwise Selection'] = {
        'Selected Features': selected_features_stepwise,
        'R Squared': r_squared_stepwise,
        'Adjusted R Squared': adj_r_squared_stepwise
    }

    # Display the feature selection results
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.max_colwidth', None):
        print(metrics_df)


def run_auto_mpg():
    auto_mpg_data = auto_mpg.get_dataset().drop(columns='brand')

    print("Feature Selection for Auto MPG")
    execute(auto_mpg_data.copy(), 'mpg')


def run_forest_fires():
    forest_fires_data = forestfires.get_dataset()

    print("Feature Selection for Forest Fires")
    execute(forest_fires_data.copy(), 'area')


def run_seoul_bike():
    seoul_bikes = seoul_bike.get_dataset()

    print("Feature Selection for Seoul Bike Sharing")
    execute(seoul_bikes.copy(), 'Rented Bike Count')


def run_boston_housing():
    boston_data = boston.get_dataset()

    print("Feature Selection for Boston Housing")
    execute(boston_data.copy(), 'MEDV')


def run():
    run_auto_mpg()
    run_forest_fires()
    run_seoul_bike()
    run_boston_housing()


run()
