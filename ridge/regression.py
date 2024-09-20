
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    rmse_ridge = mean_squared_error(y_test, y_ridge_predict, squared=False)
    rsquared_ridge = r2_score(y_test, y_ridge_predict)
    
    print("Test RMSE:", rmse_ridge)
    print("Test R-squared:", rsquared_ridge)
    
    predvactual(y_test, y_ridge_predict, 'Ridge Regression')
    
    return r_model