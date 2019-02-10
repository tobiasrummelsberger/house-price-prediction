from sklearn.model_selection import GridSearchCV
import numpy as np

def negative_mean_absolute_percentage_error(estimator, X, y_true):
    y_pred = estimator.predict(X)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return -(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def negative_root_mean_squared_error(estimator, X, y_true):
    y_pred = estimator.predict(X)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return -np.sqrt(((y_pred - y_true) ** 2).mean())

def train_gridsearch_cv(model, X_train, y_train, param_grid={}):

    clf = GridSearchCV(cv=5,
                       estimator=model,
                       param_grid=param_grid,
                       n_jobs=1,
                       verbose=1,
                       scoring='neg_mean_squared_error')
    clf.fit(X=X_train, y=y_train)

    print("MSE", -clf.best_score_)

    return clf.best_estimator_