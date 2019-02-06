from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
import pandas as pd
import math
from sklearn.decomposition import PCA

def impute_missing_values(X_train, X_test, path):
    categorical_features = list(X_train.select_dtypes(include=['object']))
    numerical_features = list(X_train.select_dtypes(include=['integer', 'float']))

    imputer_path = str(path+'imputer_{feature}.pkl')

    imputer = SimpleImputer(strategy='median')

    for feature in categorical_features:
        most_frequent_value = X_train[feature].value_counts()
        X_train[feature].replace(to_replace=np.nan, value=most_frequent_value, inplace=True)

    imputer.fit(X_train[numerical_features])
    X_train.drop(columns=numerical_features)
    X_train_imputed = pd.DataFrame(data=imputer.transform(X_train[numerical_features]), index=X_train.index, columns=numerical_features)
    X_train = pd.concat([X_train, X_train_imputed], axis=1)

    return X_train, X_test

def one_hot_encoding(X_train, X_test, path):
    categorical_features = list(X_train.select_dtypes(include=['object']))

    binarizer_path = str(path+'obj/binarizer/binarizer_{feature}.pkl')

    # train binarizer
    for feature in categorical_features:
        binarizer = LabelBinarizer()
        binarizer.fit(X_train[feature])
        joblib.dump(binarizer, binarizer_path.format(feature=feature))

    for feature in categorical_features:
        binarizer = joblib.load(binarizer_path.format(feature=feature))
        columns = [feature+'_'+class_ for class_ in binarizer.classes_]
        if len(binarizer.classes_) > 2:
            binarized_values = pd.DataFrame(data=binarizer.transform(X_train[feature]),
                                            index=X_train.index,
                                            columns=columns)
        else:
            binarized_values = pd.DataFrame(data=binarizer.transform(X_train[feature]),
                                            index=X_train.index,
                                            columns=[feature+'_'])
        X_train = pd.concat([X_train, binarized_values], axis=1)
        X_train.drop(columns=feature, inplace=True)

    for feature in categorical_features:
        binarizer = joblib.load(binarizer_path.format(feature=feature))
        columns = [feature + '_' + class_ for class_ in binarizer.classes_]
        if len(binarizer.classes_) > 2:
            data=binarizer.transform(X_test[feature])
            binarized_values = pd.DataFrame(data=binarizer.transform(X_test[feature]),
                                            index=X_test.index,
                                            columns=columns)
        else:
            binarized_values = pd.DataFrame(data=binarizer.transform(X_test[feature]),
                                            index=X_test.index,
                                            columns=[feature+'_'])
        X_test = pd.concat([X_test, binarized_values], axis=1)
        X_test.drop(columns=feature, inplace=True)

    return X_train, X_test

def principal_component_analysis(X_train, X_test, path):
    features = [feature[:feature.find('_')] for feature in list(X_train)]
    features = set(features)

    pca_path = str(path+'obj/principal_component_analysis/pca_{feature}.pkl')

    pca_feature_dict = dict()

    for feature in features:
        feature_classes = [class_ for class_ in list(X_train) if class_[:len(feature)] == feature]
        if len(feature_classes) > 3:
            pca_feature_dict[feature] = feature_classes
            #pca = PCA(n_components='mle', svd_solver='full')
            pca = PCA()
            pca.fit(X_train[feature_classes])
            joblib.dump(pca, pca_path.format(feature=feature))

    for feature in pca_feature_dict:
        pca = joblib.load(pca_path.format(feature=feature))
        pca_features = pd.DataFrame(data=pca.transform(X_train[pca_feature_dict[feature]]),
                                        index=X_train.index)
                                        #columns=[feature+'_'+class_ for class_ in list(pca.transform(X_train[pca_feature_dict[feature]]))])
        X_train = pd.concat([X_train, pca_features], axis=1)

    for feature in pca_feature_dict:
        pca = joblib.load(pca_path.format(feature=feature))
        pca_features = pd.DataFrame(data=pca.transform(X_test[pca_feature_dict[feature]]),
                                        index=X_test.index)
                                        #columns=[feature+'_'+class_ for class_ in list(pca.transform(X_train[pca_feature_dict[feature]]))])
        X_test = pd.concat([X_test, pca_features], axis=1)

    return X_train, X_test

def preprocess_alley(X_train, X_test):
    is_alley = lambda x: 1 if x is str else 0
    X_train['Alley'].fillna(value=0, inplace=True)
    X_train['Alley'] = X_train['Alley'].apply(is_alley)
    X_test['Alley'].fillna(value=0, inplace=True)
    X_test['Alley'] = X_test['Alley'].apply(is_alley)

    return X_train, X_test

def preprocess_MSZoning(X_train, X_test):
    X_train['MSZoning'].fillna(value='None', inplace=True)
    X_test['MSZoning'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_LotFrontage(X_train, X_test):
    LotFrontage_by_Neighborhood = X_train.groupby(['Neighborhood'])['LotFrontage'].median()
    LotFrontage_by_Neighborhood = pd.DataFrame(data=LotFrontage_by_Neighborhood)

    X_train.drop(columns=['LotFrontage'], inplace=True)
    X_train = X_train.join(other=LotFrontage_by_Neighborhood,
                           on='Neighborhood',
                           how='left')

    X_test.drop(columns=['LotFrontage'], inplace=True)
    X_test = X_test.join(other=LotFrontage_by_Neighborhood,
                           on='Neighborhood',
                           how='left')


    return X_train, X_test

def preprocess_MasVnrType(X_train, X_test):
    X_train['MasVnrType'].fillna(value='None', inplace=True)
    X_test['MasVnrType'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_BsmtQual(X_train, X_test):
    X_train['BsmtQual'].fillna(value='None', inplace=True)
    X_test['BsmtQual'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_BsmtCond(X_train, X_test):
    X_train['BsmtCond'].fillna(value='None', inplace=True)
    X_test['BsmtCond'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_BsmtExposure(X_train, X_test):
    X_train['BsmtExposure'].fillna(value='None', inplace=True)
    X_test['BsmtExposure'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_BsmtFinType1(X_train, X_test):
    X_train['BsmtFinType1'].fillna(value='None', inplace=True)
    X_test['BsmtFinType1'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_BsmtFinType2(X_train, X_test):
    X_train['BsmtFinType2'].fillna(value='None', inplace=True)
    X_test['BsmtFinType2'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_Electrical(X_train, X_test):
    most_frequent_electrical_system = X_train['Electrical'].value_counts().keys()[0]
    X_train['Electrical'].fillna(value=most_frequent_electrical_system, inplace=True)
    X_test['Electrical'].fillna(value=most_frequent_electrical_system, inplace=True)

    return X_train, X_test

def preprocess_FireplaceQu(X_train, X_test):
    X_train['FireplaceQu'].fillna(value='None', inplace=True)
    X_test['FireplaceQu'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_GarageType(X_train, X_test):
    X_train['GarageType'].fillna(value='None', inplace=True)
    X_test['GarageType'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_GarageFinish(X_train, X_test):
    X_train['GarageFinish'].fillna(value='None', inplace=True)
    X_test['GarageFinish'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_GarageQual(X_train, X_test):
    X_train['GarageQual'].fillna(value='None', inplace=True)
    X_test['GarageQual'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_GarageCond(X_train, X_test):
    X_train['GarageCond'].fillna(value='None', inplace=True)
    X_test['GarageCond'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_PoolQC(X_train, X_test):
    X_train['PoolQC'].fillna(value='None', inplace=True)
    X_test['PoolQC'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_Fence(X_train, X_test):
    X_train['Fence'].fillna(value='None', inplace=True)
    X_test['Fence'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_MiscFeature(X_train, X_test):
    X_train['MiscFeature'].fillna(value='None', inplace=True)
    X_test['MiscFeature'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_Utilities(X_train, X_test):
    X_train['Utilities'].fillna(value='None', inplace=True)
    X_test['Utilities'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_Exterior1st(X_train, X_test):
    X_train['Exterior1st'].fillna(value='None', inplace=True)
    X_test['Exterior1st'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_Exterior2nd(X_train, X_test):
    X_train['Exterior2nd'].fillna(value='None', inplace=True)
    X_test['Exterior2nd'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_KitchenQual(X_train, X_test):
    X_train['KitchenQual'].fillna(value='None', inplace=True)
    X_test['KitchenQual'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_Functional(X_train, X_test):
    X_train['Functional'].fillna(value='None', inplace=True)
    X_test['Functional'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_SaleType(X_train, X_test):
    X_train['SaleType'].fillna(value='None', inplace=True)
    X_test['SaleType'].fillna(value='None', inplace=True)

    return X_train, X_test

def preprocess_GarageYrBlt(X_train, X_test):
    GarageYrBlt_mean = X_train['GarageYrBlt'].mean()
    X_train['GarageYrBlt'].fillna(value=GarageYrBlt_mean, inplace=True)
    X_test['GarageYrBlt'].fillna(value=GarageYrBlt_mean, inplace=True)

    return X_train, X_test

def preprocess_MasVnrArea(X_train, X_test):
    MasVnrArea_mean = X_train[X_train['MasVnrType'] != 'None']['MasVnrArea'].mean()
    X_train['MasVnrArea'].apply(lambda x: x if x is not None else MasVnrArea_mean)
    X_test['MasVnrArea'].apply(lambda x: x if x is not None else MasVnrArea_mean)

    return X_train, X_test