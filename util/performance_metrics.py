import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, explained_variance_score, \
    median_absolute_error, mean_absolute_error

def get_r2_score(y_true, y_pred):
    return r2_score(y_true=y_true, y_pred=y_pred)

def get_mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)

def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

def get_all(y_true, y_pred):
    mse = round(get_mse(y_true, y_pred), 5)
    r2 = round(get_r2_score(y_true, y_pred), 5)
    rmse = round(get_rmse(y_true, y_pred), 5)

    return {
        'MSE': f"{mse:.5f}",
        'R²': f"{r2:.5f}",
        'RMSE': rmse,
    }