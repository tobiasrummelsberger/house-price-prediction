import numpy as np

def negative_mean_absolute_percentage_error(estimator, X, y_true):
    y_pred = estimator.predict(X)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return -(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)