import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from util import performance_metrics


def random_forest(X, y, X_train, X_test, y_train, y_test, result):
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train.values.ravel())
    y_pred_in_sample_rf = rf_model.predict(X)
    result["Random Forest (In-Sample)"] = performance_metrics.get_all(y, y_pred_in_sample_rf)

    y_train_pred_rf = rf_model.predict(X_train)
    result["Random Forest (Train)"] = performance_metrics.get_all(y_train, y_train_pred_rf)
    y_test_pred_rf = rf_model.predict(X_test)
    result["Random Forest (Test)"] = performance_metrics.get_all(y_test, y_test_pred_rf)

    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("Feature Importance:")
    print(feature_importance_df)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse = -cross_val_score(rf_model, X, y.values.ravel(), cv=kf, scoring='neg_mean_squared_error')
    r2 = cross_val_score(rf_model, X, y.values.ravel(), cv=kf, scoring='r2')
    rmse = np.sqrt(mse)
    result["Random Forest (5X Validation)"] = {
        'MSE': f"{np.mean(mse):.5f}",
        'R²': f"{np.mean(r2):.5f}",
        'RMSE': f"{np.mean(rmse):.5f}",
    }