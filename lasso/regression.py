import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from util import performance_metrics


def lasso(X, y, X_train, X_test, y_train, y_test, result, iter=1000):
    # In-sample Lasso Regression
    in_sample_lasso_reg = Lasso(alpha=0.001, max_iter=iter)
    in_sample_lasso_reg.fit(X, y)
    y_pred = in_sample_lasso_reg.predict(X)
    result["Lasso Regression (In-Sample)"] = performance_metrics.get_all(y, y_pred)

    # Lasso Regression on Train and Test sets
    lasso_reg = Lasso(alpha=0.1, max_iter=iter)
    lasso_reg.fit(X_train, y_train)
    y_train_pred = lasso_reg.predict(X_train)
    y_test_pred = lasso_reg.predict(X_test)
    result["Lasso Regression (Train)"] = performance_metrics.get_all(y_train, y_train_pred)
    result["Lasso Regression (Test)"] = performance_metrics.get_all(y_test, y_test_pred)

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(in_sample_lasso_reg, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(in_sample_lasso_reg, X, y, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    result["Lasso Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }

# import numpy as np
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
# from util import performance_metrics
#
#
# def lasso(X, y, X_train, X_test, y_train, y_test, result):
#     # Use GridSearchCV to find the best alpha (regularization strength)
#     alpha_values = np.logspace(-4, 0, 50)  # Test alpha values between 0.0001 and 1.0
#     lasso_reg = Lasso(max_iter=10000)
#
#     param_grid = {'alpha': alpha_values}
#
#     # Perform cross-validation with 5 splits and MSE as the scoring metric
#     grid_search = GridSearchCV(lasso_reg, param_grid, cv=5, scoring='r2', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#
#     # Best model after cross-validation
#     best_lasso_reg = grid_search.best_estimator_
#
#     # In-sample Lasso Regression
#     best_lasso_reg.fit(X, y)
#     y_pred = best_lasso_reg.predict(X)
#     result["Lasso Regression (In-Sample)"] = performance_metrics.get_all(y, y_pred)
#
#     # Lasso Regression on Train and Test sets
#     y_train_pred = best_lasso_reg.predict(X_train)
#     y_test_pred = best_lasso_reg.predict(X_test)
#     result["Lasso Regression (Train)"] = performance_metrics.get_all(y_train, y_train_pred)
#     result["Lasso Regression (Test)"] = performance_metrics.get_all(y_test, y_test_pred)
#
#     # 5-Fold Cross Validation for best alpha
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     mse = -cross_val_score(best_lasso_reg, X, y, cv=kf, scoring='neg_mean_squared_error')
#     r2 = cross_val_score(best_lasso_reg, X, y, cv=kf, scoring='r2')
#     rmse = np.sqrt(mse)
#     result["Lasso Regression (5X Validation)"] = {
#         'MSE': f"{np.mean(mse):.5f}",
#         'R²': f"{np.mean(r2):.5f}",
#         'RMSE': f"{np.mean(rmse):.5f}",
#     }
#
#     print(f"Best alpha value: {grid_search.best_params_['alpha']}")
#     print(f"Best R² score from CV: {grid_search.best_score_}")
#
