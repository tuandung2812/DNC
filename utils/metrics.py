import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error


def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_pred-y_true)**2))


def MAPE(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


metrics = {
    "RMSE": RMSE,
    "MAPE": MAPE,
    "R2": R2,
    "MAE": MAE
}
