
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score

from util import performance_metrics


def predvactual(y_test, y_test_pred, model_name):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test, y=y_test_pred, color="orange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='pink')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.show()

def ridge_regression_model(x_train, x_test, y_train, y_test, alpha=1):
    r_model = Ridge(alpha=1.0)
    r_model.fit(x_train, y_train)
    
    y_ridge_predict = r_model.predict(x_test)

    result = performance_metrics.get_all(y_test, y_ridge_predict)
    train_res = performance_metrics.get_all(y_test, y_ridge_predict)
    print("Train Scores", train_res)
    print("Test Scores", result)
    
    predvactual(y_test, y_ridge_predict, 'Ridge Regression')
    
    return r_model


def cross_validation_scores(lr, x, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(lr, x, y, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(lr, x, y, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    return {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }