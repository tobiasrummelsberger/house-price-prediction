from sklearn.model_selection import GridSearchCV
import numpy as np

def negative_mean_absolute_percentage_error(estimator, X, y_true):
    y_pred = estimator.predict(X)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return -(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def train_gridsearch_cv(model, X_train, y_train, param_grid={}):

    clf = GridSearchCV(cv=3,
                       estimator=model,
                       param_grid=param_grid,
                       n_jobs=-1,
                       verbose=1)
    clf.fit(X=X_train, y=y_train)

    return clf.best_estimator_