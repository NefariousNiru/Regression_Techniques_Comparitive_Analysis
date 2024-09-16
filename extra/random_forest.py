import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from util import performance_metrics


def random_forest(X, y, X_train, X_test, y_train, y_test, result):
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train.values.ravel())
    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)
    result["Random Forest (Train)"] = performance_metrics.get_all(y_train, y_train_pred_rf)
    result["Random Forest (Test)"] = performance_metrics.get_all(y_test, y_test_pred_rf)

    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df)