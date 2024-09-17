import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def forward_selection(X, y):
    y = np.ravel(y)  # Convert y to a 1D array
    remaining_features = list(X.columns)
    selected_features = []
    best_r_squared = 0
    best_model = None

    while remaining_features:
        scores_with_candidates = []
        for candidate in remaining_features:
            X_new = X[selected_features + [candidate]]
            model = LinearRegression().fit(X_new, y)
            score = r2_score(y, model.predict(X_new))
            scores_with_candidates.append((score, candidate))

        if not scores_with_candidates:
            break

        scores_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = scores_with_candidates[0]

        if best_new_score > best_r_squared:
            selected_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            best_r_squared = best_new_score
            best_model = LinearRegression().fit(X[selected_features], y)
        else:
            break

    # Return both R Squared and Adjusted R Squared
    if best_model:
        n = len(y)
        p = len(selected_features)
        adj_r_squared = 1 - (1 - best_r_squared) * (n - 1) / (n - p - 1)
        return selected_features, best_r_squared, adj_r_squared
    return selected_features, None, None


def backward_selection(X, y, significance_level=0.05):
    y = np.ravel(y)  # Convert y to a 1D array
    selected_features = list(X.columns)
    while len(selected_features) > 0:
        X_new = X[selected_features]
        model = LinearRegression().fit(X_new, y)
        p_values = f_regression(X_new, y)[1]  # Get p-values for each feature

        if len(p_values) == 0:
            break

        max_p_value = np.max(p_values)

        if max_p_value > significance_level:
            feature_to_remove = selected_features[np.argmax(p_values)]
            selected_features.remove(feature_to_remove)
        else:
            break

    # Return both R Squared and Adjusted R Squared
    if selected_features:
        X_new = X[selected_features]
        model = LinearRegression().fit(X_new, y)
        r_squared = r2_score(y, model.predict(X_new))
        n = len(y)
        p = len(selected_features)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        return selected_features, r_squared, adj_r_squared
    return [], None, None


def stepwise_selection(X, y, significance_level_in=0.05, significance_level_out=0.05):
    y = np.ravel(y)  # Convert y to a 1D array
    initial_features = X.columns.tolist()
    selected_features = []
    best_model = None

    while True:
        # Forward Selection
        remaining_features = list(set(initial_features) - set(selected_features))
        if not remaining_features:
            break

        new_p_values = pd.Series(index=remaining_features)

        for new_column in remaining_features:
            X_new = X[selected_features + [new_column]]
            model = LinearRegression().fit(X_new, y)
            p_value = f_regression(X_new, y)[1][-1]  # Get p-value for the last added column
            new_p_values[new_column] = p_value

        best_p_value = new_p_values.min()
        if best_p_value < significance_level_in:
            best_feature = new_p_values.idxmin()
            selected_features.append(best_feature)
            best_model = LinearRegression().fit(X[selected_features], y)
        else:
            break

        # Backward Elimination
        if len(selected_features) > 0:
            X_new = X[selected_features]
            p_values = f_regression(X_new, y)[1]
            if len(p_values) == 0:
                break

            worst_p_value = np.max(p_values)
            if worst_p_value > significance_level_out:
                worst_feature = selected_features[np.argmax(p_values)]
                selected_features.remove(worst_feature)
                best_model = LinearRegression().fit(X[selected_features], y)
            else:
                break

    # Return both R Squared and Adjusted R Squared
    if best_model:
        r_squared = r2_score(y, best_model.predict(X[selected_features]))
        n = len(y)
        p = len(selected_features)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        return selected_features, r_squared, adj_r_squared
    return selected_features, None, None