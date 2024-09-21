from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PowerTransformer
import numpy as np
from util import performance_metrics

'''
Performs log transform on y
'''
def logarithmic(X, y, X_train, X_test, y_train, y_test, result):
    y_log = np.log1p(y)
    in_sample_log_reg = LinearRegression()
    in_sample_log_reg.fit(X, y_log)
    y_pred_in_sample_log = np.expm1(in_sample_log_reg.predict(X))
    result["Logarithmic Regression (In-Sample)"] = performance_metrics.get_all(y, y_pred_in_sample_log)

    # Train Test
    y_train_log = np.log1p(y_train)
    log_reg = LinearRegression()
    log_reg.fit(X_train, y_train_log)
    y_train_pred_log = np.expm1(log_reg.predict(X_train))
    y_test_pred_log = np.expm1(log_reg.predict(X_test))
    result["Logarithmic Regression (Train)"] = performance_metrics.get_all(y_train, y_train_pred_log)
    result["Logarithmic Regression (Test)"] = performance_metrics.get_all(y_test, y_test_pred_log)

    performance_metrics.predvactual(np.squeeze(y_test), np.squeeze(y_test_pred_log), "Logarithmic")

    # 5 - Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(in_sample_log_reg, X, np.log1p(y), cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(in_sample_log_reg, X, np.log1p(y), cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    result["Logarithmic Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }

'''
Performs square-root transform on y
'''
def square_root(X, y, X_train, X_test, y_train, y_test, result):
    in_sample_reg = LinearRegression()
    in_sample_reg.fit(X, y)
    y_pred_in_sample = in_sample_reg.predict(X)
    result["Square Root Regression (In-Sample)"] = performance_metrics.get_all(y, y_pred_in_sample)

    # Train Test
    y_train_sqrt = np.sqrt(y_train)
    sqrt_reg = LinearRegression()
    sqrt_reg.fit(X_train, y_train_sqrt)
    y_train_pred_sqrt = sqrt_reg.predict(X_train) ** 2
    y_test_pred_sqrt = sqrt_reg.predict(X_test) ** 2
    result["Square Root Regression (Train)"] = performance_metrics.get_all(y_train, y_train_pred_sqrt)
    result["Square Root Regression (Test)"] = performance_metrics.get_all(y_test, y_test_pred_sqrt)

    performance_metrics.predvactual(np.squeeze(y_test), np.squeeze(y_test_pred_sqrt), "Square Root")

    # 5 - Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(in_sample_reg, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(in_sample_reg, X, y, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    result["Square Root Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }

'''
Performs reciprocal transform on y
'''
def reciprocal(X, y, X_train, X_test, y_train, y_test, result):
    y_reciprocal = 1 / (y + 1e-9)
    in_sample_reciprocal_reg = LinearRegression()
    in_sample_reciprocal_reg.fit(X, y_reciprocal)
    y_pred_in_sample_reciprocal = 1 / in_sample_reciprocal_reg.predict(X)
    result["Reciprocal Regression (In-Sample)"] = performance_metrics.get_all(y, y_pred_in_sample_reciprocal)

    # Train Test
    y_train_reciprocal = 1 / (y_train + 1e-9)
    reciprocal_reg = LinearRegression()
    reciprocal_reg.fit(X_train, y_train_reciprocal)
    y_train_pred_reciprocal = 1 / reciprocal_reg.predict(X_train)
    y_test_pred_reciprocal = 1 / reciprocal_reg.predict(X_test)
    result["Reciprocal Regression (Train)"] = performance_metrics.get_all(y_train, y_train_pred_reciprocal)
    result["Reciprocal Regression (Test)"] = performance_metrics.get_all(y_test, y_test_pred_reciprocal)

    performance_metrics.predvactual(np.squeeze(y_test), np.squeeze( y_test_pred_reciprocal), "Reciprocal")

    # 5 - Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(in_sample_reciprocal_reg, X, 1 / (y + 1e-9), cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(in_sample_reciprocal_reg, X, 1 / (y + 1e-9), cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    result["Reciprocal Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }

'''
Performs yeo-johnson box cox transform on y
'''
def box_cox(X, y, X_train, X_test, y_train, y_test, result):
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    yeo_johnson_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    y_transformed = yeo_johnson_transformer.fit_transform(np.array(y).reshape(-1, 1)).flatten()
    in_sample_yeo_johnson_reg = LinearRegression()
    in_sample_yeo_johnson_reg.fit(X, y_transformed)
    y_pred_in_sample_transformed = in_sample_yeo_johnson_reg.predict(X).reshape(-1, 1)
    y_pred_in_sample = yeo_johnson_transformer.inverse_transform(y_pred_in_sample_transformed).flatten()
    result["Yeo-Johnson Regression (In-Sample)"] = performance_metrics.get_all(y, y_pred_in_sample)

    # Train Test
    y_train_transformed = yeo_johnson_transformer.fit_transform(y_train).flatten()
    yeo_johnson_reg = LinearRegression()
    yeo_johnson_reg.fit(X_train, y_train_transformed)
    y_train_pred_transformed = yeo_johnson_reg.predict(X_train).reshape(-1, 1)
    y_train_pred = yeo_johnson_transformer.inverse_transform(y_train_pred_transformed).flatten()
    y_test_pred_transformed = yeo_johnson_reg.predict(X_test).reshape(-1, 1)
    y_test_pred = yeo_johnson_transformer.inverse_transform(y_test_pred_transformed).flatten()
    result["Yeo-Johnson Regression (Train)"] = performance_metrics.get_all(y_train.flatten(), y_train_pred)
    result["Yeo-Johnson Regression (Test)"] = performance_metrics.get_all(y_test.flatten(), y_test_pred)

    performance_metrics.predvactual(np.squeeze(y_test), np.squeeze(y_test_pred), "Box Cox")

    # 5 - Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(in_sample_yeo_johnson_reg, X, y_transformed, cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(in_sample_yeo_johnson_reg, X, y_transformed, cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    result["Yeo-Johnson Regression (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }
