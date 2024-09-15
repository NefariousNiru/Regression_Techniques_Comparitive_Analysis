from sklearn.metrics import r2_score, mean_squared_error

def get_r2_score(y_true, y_pred):
    return r2_score(y_true=y_true, y_pred=y_pred)

def get_mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)