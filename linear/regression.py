import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from util import performance_metrics

def predvactual(y_test, y_test_pred, model_name):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test, y=y_test_pred, color="orange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='pink')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.show()

def linear_regression_model(x_train, x_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)

    result = performance_metrics.get_all(y_test, y_test_pred)
    train_res = performance_metrics.get_all(y_test, y_test_pred)
    print("Train Scores", train_res)
    print("Test Scores", result)
    
    predvactual(y_test, y_test_pred, 'Linear Regression')
    return lr


