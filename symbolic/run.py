import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datasets.auto_mpg import auto_mpg
from datasets.forest_fires import forestfires
from datasets.seoul_bike_sharing_demand import seoul_bike
from datasets.boston import boston

# Function to calculate adjusted R-squared
def adjusted_r_squared(r_squared, n, p):
    return 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Function to perform symbolic regression and evaluate quality of fit
def symbolic_regression(data, target_column):
    # Features (X) and target (y)
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    # Define the symbolic regressor
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=30,
                               tournament_size=20,
                               stopping_criteria=0.001,
                               const_range=(-1.0, 1.0),
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.01,
                               random_state=0)

    # In-Sample fit: Train the model on the full dataset
    est_gp.fit(X, y)

    # Predict on the same dataset (In-Sample)
    y_in_sample_pred = est_gp.predict(X)
    mse_in_sample = mean_squared_error(y, y_in_sample_pred)
    r2_in_sample = r2_score(y, y_in_sample_pred)
    adj_r2_in_sample = adjusted_r_squared(r2_in_sample, len(y), X.shape[1])
    mae_in_sample = mean_absolute_error(y, y_in_sample_pred)

    print(f'In-Sample Mean Squared Error: {mse_in_sample}')
    print(f'In-Sample R-squared: {r2_in_sample}')
    print(f'In-Sample Adjusted R-squared: {adj_r2_in_sample}')
    print(f'In-Sample Mean Absolute Error: {mae_in_sample}')

    # Train-Test Split (80-20 validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model on training data
    est_gp.fit(X_train, y_train)

    # Predict on test data and calculate metrics
    y_val_pred = est_gp.predict(X_test)
    mse_val = mean_squared_error(y_test, y_val_pred)
    r2_val = r2_score(y_test, y_val_pred)
    adj_r2_val = adjusted_r_squared(r2_val, len(y_test), X_test.shape[1])
    mae_val = mean_absolute_error(y_test, y_val_pred)

    print(f'Validation Mean Squared Error: {mse_val}')
    print(f'Validation R-squared: {r2_val}')
    print(f'Validation Adjusted R-squared: {adj_r2_val}')
    print(f'Validation Mean Absolute Error: {mae_val}')

    # 5x Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(est_gp, X, y, cv=kf, scoring='neg_mean_squared_error')
    cross_val_mse = -1 * np.mean(cross_val_scores)
    print(f'5x Cross-Validation Mean Squared Error: {cross_val_mse}')

    # Plot In-Sample predicted vs actual
    plt.scatter(y, y_in_sample_pred, edgecolor='k', facecolor='none', alpha=0.7)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('In-Sample: Predicted vs Actual')
    plt.show()

    # Print the best symbolic expression found
    print(f'Best expression: {est_gp._program}')

# Function to run symbolic regression on all datasets
def run_all_datasets():
    # Auto MPG dataset
    print("Auto MPG Dataset")
    auto_mpg_data = auto_mpg.get_dataset().drop(columns='brand')  # Adjust for correct target
    symbolic_regression(auto_mpg_data, target_column='mpg')  # Example target column for auto_mpg

    # Forest Fires dataset
    print("\nForest Fires Dataset")
    forest_fires_data = forestfires.get_dataset()
    symbolic_regression(forest_fires_data, target_column='area')  # Example target column for forest fires

    # Seoul Bike Sharing dataset
    print("\nSeoul Bike Sharing Dataset")
    seoul_bike_data = seoul_bike.get_dataset()
    symbolic_regression(seoul_bike_data, target_column='rented_bike_count')  # Example target column for Seoul bike

    # Boston Housing dataset
    print("\nBoston Housing Dataset")
    boston_data = boston.get_dataset()
    symbolic_regression(boston_data, target_column='medv')  # Example target column for Boston dataset

# Run symbolic regression on all datasets
run_all_datasets()
