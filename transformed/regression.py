from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from util import performance_metrics

def linear(X_train, X_test, y_train, y_test, results):
    # Linear Regression (Baseline)
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred_linear = linear_reg.predict(X_test)
    results["Linear Regression (Baseline)"] = performance_metrics.get_all(y_test, y_pred_linear)

def polynomial(X_train, X_test, y_train, y_test, result, degree=2):
    # Polynomial Regression (Degree 2)
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    y_pred_poly = poly_reg.predict(X_test_poly)
    result["Polynomial Regression (Degree 2)"] = performance_metrics.get_all(y_test, y_pred_poly)

def logarithmic(X_train, X_test, y_train, y_test, result):
    # Logarithmic Transformation (apply log to the features)
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)
    log_reg = LinearRegression()
    log_reg.fit(X_train_log, y_train)
    y_pred_log = log_reg.predict(X_test_log)
    result["Logarithmic Regression"] = performance_metrics.get_all(y_test, y_pred_log)





