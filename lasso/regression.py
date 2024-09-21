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

    performance_metrics.predvactual(np.squeeze(y_test), y_test_pred, "Lasso")

