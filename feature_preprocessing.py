from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
import pandas as pd


def one_hot_encoding(X_train, X_test, path):
    categorical_features = list(X_train.select_dtypes(include=['object']))

    binarizer_path = str(path+'binarizer{feature}.pkl')

    # train binarizer
    for feature in categorical_features:
        binarizer = MultiLabelBinarizer()
        print(X_train[feature])
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

