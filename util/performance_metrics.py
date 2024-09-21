import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

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

def predvactual(y_test, y_test_pred, model_name):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test, y=y_test_pred, color="orange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='pink')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.show()

