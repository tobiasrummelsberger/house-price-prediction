import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_path, train_key='SalePrice'):
    data = pd.read_csv(train_path)
    y = data[train_key]
    X = data.drop(columns=train_key)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    categorical_features = list(X.select_dtypes(include=['object']))
    numerical_features = list(X.select_dtypes(include=['integer', 'float']))

    print("The data set contains {number_categorical_features} categorical features and {number_numerical_features}"
          " numerical features. The train data set contains {train_items} rows and the test data set contains "
          "{test_items} rows.".format(number_categorical_features=len(categorical_features),
                                      number_numerical_features=len(numerical_features),
                                      train_items = X_train.shape[1],
                                      test_items = X_test.shape[1]
                                      ))
    print("The categorical features are {categorical_features} and the numerical features are "
          "{numerical_features}".format(categorical_features=str(categorical_features),
                                        numerical_features=str(numerical_features)))

    return X_train, X_test, y_train, y_test

