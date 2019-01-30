from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
import pandas as pd

def depr_impute_missing_values(X_train, X_test, path):
    categorical_features = list(X_train.select_dtypes(include=['object']))
    numerical_features = list(X_train.select_dtypes(include=['integer', 'float']))

    imputer_path = str(path+'imputer_{feature}.pkl')

    ## categorical features

    for feature in categorical_features:
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(X_train[feature])
        joblib.dump(imputer, imputer_path.format(feature=feature))

    for feature in categorical_features:
        imputer = joblib.load(imputer_path.format(feature=feature))
        X_train[feature] = imputer.fit(X_train[feature])

    for feature in categorical_features:
        imputer = joblib.load(imputer_path.format(feature=feature))
        X_test[feature] = imputer.fit(X_test[feature])

    return X_train, X_test

def impute_missing_values(X_train, X_test, path):
    categorical_features = list(X_train.select_dtypes(include=['object']))
    numerical_features = list(X_train.select_dtypes(include=['integer', 'float']))

    imputer_path = str(path+'imputer_{feature}.pkl')

    imputer = SimpleImputer(strategy='most_frequent')

    imputer.fit(X_train[categorical_features])
    X_train.drop(columns=categorical_features)
    X_train_imputed = pd.DataFrame(data=imputer.transform(X_train), index=X_train.index, columns=categorical_features)
    X_train = pd.concat([X_train, X_train_imputed], axis=1)

    return X_train, X_test



def one_hot_encoding(X_train, X_test, path):
    categorical_features = list(X_train.select_dtypes(include=['object']))

    binarizer_path = str(path+'binarizer_{feature}.pkl')

    # train binarizer
    for feature in categorical_features:
        binarizer = MultiLabelBinarizer()
        binarizer.fit(X_train[feature])
        joblib.dump(binarizer, binarizer_path.format(feature=feature))

    for feature in categorical_features:
        binarizer = joblib.load(binarizer_path.format(feature=feature))
        if len(binarizer.classes) > 1:
            binarized_values = pd.DataFrame(data=binarizer.transform(X_train[feature]),
                                            index=X_train.index,
                                            columns=binarizer.classes)
        else:
            binarized_values = pd.DataFrame(data=binarizer.transform(X_train[feature]),
                                            index=X_train.index,
                                            columns=feature)
        X_train = pd.concat([X_train, binarized_values], axis=1)
        X_train.drop(columns=feature)

    for feature in categorical_features:
        binarizer = joblib.load(binarizer_path.format(feature=feature))
        if len(binarizer.classes) > 1:
            binarized_values = pd.DataFrame(data=binarizer.transform(X_test[feature]),
                                            index=X_test.index,
                                            columns=binarizer.classes)
        else:
            binarized_values = pd.DataFrame(data=binarizer.transform(X_test[feature]),
                                            index=X_test.index,
                                            columns=feature)
        X_test = pd.concat([X_test, binarized_values], axis=1)
        X_test.drop(columns=feature)

    return X_train, X_test

