import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datasets.auto_mpg import auto_mpg


# Drop rows with missing values for a clean dataset
auto_mpg_data = auto_mpg.get_dataset()

data_clean = auto_mpg_data.dropna()

# Separate target variable (y) and features (X)
X = data_clean[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']]
y = data_clean['mpg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize results dictionary to store the performance of different models
metrics = {}

def calculate_metrics(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'R²': r2}

# Linear Regression (Baseline)
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)
metrics['Linear Regression'] = calculate_metrics('Linear Regression', y_test, y_pred_linear)

# Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
metrics['Polynomial Regression (Degree 2)'] = calculate_metrics('Polynomial Regression (Degree 2)', y_test, y_pred_poly)

# Logarithmic Transformation (apply log to the features)
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)
log_reg = LinearRegression()
log_reg.fit(X_train_log, y_train)
y_pred_log = log_reg.predict(X_test_log)
metrics['Logarithmic Transformation'] = calculate_metrics('Logarithmic Transformation', y_test, y_pred_log)

# Standard Scaler Transformation (Standardized features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaled_reg = LinearRegression()
scaled_reg.fit(X_train_scaled, y_train)
y_pred_scaled = scaled_reg.predict(X_test_scaled)
metrics['Standard Scaled Features'] = calculate_metrics('Standard Scaled Features', y_test, y_pred_scaled)

# Display the metrics
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
print(metrics_df)
