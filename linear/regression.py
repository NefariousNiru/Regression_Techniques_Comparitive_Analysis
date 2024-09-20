from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
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

def linear_regression_model(x_train, x_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    
    rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    rsquared = r2_score(y_train, y_train_pred)
    
    valid_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    valid_rsquared = r2_score(y_test, y_test_pred)
    
    print("Train RMSE:", rmse)
    print("Train R-squared:", rsquared)
    print("Test RMSE:", valid_rmse)
    print("Test R-squared:", valid_rsquared)
    
    predvactual(y_test, y_test_pred, 'Linear Regression')
    return lr

def cross_validation_scores(lr, x, y):
    crossval_scores = cross_val_score(lr, x, y, cv=5)
    cv_rmse = (-crossval_scores.mean())**0.5
    return cv_rmse