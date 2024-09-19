import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def adjusted_r_squared(r_squared, n, p):
    """
    Calculate adjusted R-squared.
    :param r_squared: The R-squared value
    :param n: Number of observations
    :param p: Number of predictors
    :return: Adjusted R-squared value
    """
    return 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Function to perform symbolic regression on a dataset and evaluate quality of fit
def symbolic_regression(file_path, target_column):
    # Load the dataset, making sure to handle non-numeric columns like 'Date'
    data = pd.read_csv(file_path)

    # Drop rows with NaN values (you could handle missing values differently if preferred)
    data.dropna(inplace=True)

    # Convert only the numeric columns to proper numeric types
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # One-hot encode categorical columns
    data = pd.get_dummies(data, drop_first=True)

    # Drop rows that may have been coerced to NaN after conversion
    data.dropna(inplace=True)

    # Ensure the target column is numeric
    if target_column not in numeric_columns:
        raise ValueError(f'Target column "{target_column}" is not numeric.')

    # Features (X) and target (y)
    X = data.drop(columns=target_column).values
    y = data[target_column].values

    # In-Sample Fit
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

# Replace with your dataset file path relative to the project directory
file_path = './data/your_dataset.csv'

# Replace with your target column name
target_column = "target_column_name"

# Perform symbolic regression
symbolic_regression(file_path, target_column)
