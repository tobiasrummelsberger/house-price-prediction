from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def one_hot_encoding(*datasets):
    categorical_features = list(datasets[0].select_dtypes(include=['object']))
    path = '/Users/tobias/PycharmProjects/house-price-prediction'
    # train binarizer
    for feature in categorical_features:
        binarizer_path = str(path + '/obj/binarizer/binarizer_{feature}.pkl')
        binarizer = LabelBinarizer()
        binarizer.fit(datasets[0][feature])
        joblib.dump(binarizer, binarizer_path.format(feature=feature))

    for dataset in datasets:
        for feature in categorical_features:
            binarizer = joblib.load(binarizer_path.format(feature=feature))
            columns = [feature+'_'+class_ for class_ in binarizer.classes_]
            if len(binarizer.classes_) > 2:
                binarized_values = pd.DataFrame(data=binarizer.transform(dataset[feature]),
                                                index=dataset.index,
                                                columns=columns)
            else:
                binarized_values = pd.DataFrame(data=binarizer.transform(dataset[feature]),
                                                index=dataset.index,
                                                columns=[feature+'_'])
            dataset = pd.concat([dataset, binarized_values], axis=1)
            #dataset.drop(columns=[feature], inplace=True)

    for dataset in datasets:
        dataset.drop(columns=categorical_features, inplace=True)

    return datasets

def principal_component_analysis(path, *datasets):
    features = [feature[:feature.find('_')] for feature in list(datasets[0])]
    features = set(features)

    pca_path = str(path+'/obj/principal_component_analysis/pca_{feature}.pkl')

    pca_feature_dict = dict()
    for feature in features:
        feature_classes = [class_ for class_ in list(datasets[0]) if class_[:len(feature)] == feature]
        if len(feature_classes) > 3:
            pca_feature_dict[feature] = feature_classes
            #pca = PCA(n_components='mle', svd_solver='full')
            pca = PCA()
            pca.fit(datasets[0][feature_classes])
            joblib.dump(pca, pca_path.format(feature=feature))

    for dataset in datasets:
        for feature in pca_feature_dict:
            pca = joblib.load(pca_path.format(feature=feature))
            pca_features = pd.DataFrame(data=pca.transform(dataset[pca_feature_dict[feature]]),
                                            index=dataset.index)
                                            #columns=[feature+'_'+class_ for class_ in list(pca.transform(X_train[pca_feature_dict[feature]]))])
            dataset = pd.concat([dataset, pca_features], axis=1)

    return datasets

def preprocess_alley(*datasets):
    is_alley = lambda x: 1 if x is str else 0
    for dataset in datasets:
        dataset['Alley'].fillna(value=0, inplace=True)
        dataset['Alley'] = dataset['Alley'].apply(is_alley)
    return datasets

def preprocess_MSZoning(*datasets):
    for dataset in datasets:
        dataset['MSZoning'].fillna(value='None', inplace=True)
    return datasets

def preprocess_LotFrontage(*datasets):
    LotFrontage_by_Neighborhood = datasets[0].groupby(['Neighborhood'])['LotFrontage'].median()
    LotFrontage_by_Neighborhood = pd.DataFrame(data=LotFrontage_by_Neighborhood)
    for dataset in datasets:
        dataset.drop(columns=['LotFrontage'], inplace=True)
        dataset = dataset.join(other=LotFrontage_by_Neighborhood,
                           on='Neighborhood',
                           how='left')
    return datasets

def preprocess_MasVnrType(*datasets):
    for dataset in datasets:
        dataset['MasVnrType'].fillna(value='None', inplace=True)
    return datasets

def preprocess_BsmtQual(*datasets):
    for dataset in datasets:
        dataset['BsmtQual'].fillna(value='None', inplace=True)
    return datasets

def preprocess_BsmtCond(*datasets):
    for dataset in datasets:
        dataset['BsmtCond'].fillna(value='None', inplace=True)
    return datasets

def preprocess_BsmtExposure(*datasets):
    for dataset in datasets:
        dataset['BsmtExposure'].fillna(value='None', inplace=True)
    return datasets

def preprocess_BsmtFinType1(*datasets):
    for dataset in datasets:
        dataset['BsmtFinType1'].fillna(value='None', inplace=True)
    return datasets

def preprocess_BsmtFinType2(*datasets):
    for dataset in datasets:
        dataset['BsmtFinType2'].fillna(value='None', inplace=True)
    return datasets

def preprocess_Electrical(*datasets):
    most_frequent_electrical_system = datasets[0]['Electrical'].value_counts().keys()[0]
    for dataset in datasets:
        dataset['Electrical'].fillna(value=most_frequent_electrical_system, inplace=True)
    return datasets

def preprocess_FireplaceQu(*datasets):
    for dataset in datasets:
        dataset['FireplaceQu'].fillna(value='None', inplace=True)
    return datasets

def preprocess_GarageType(*datasets):
    for dataset in datasets:
        dataset['GarageType'].fillna(value='None', inplace=True)
    return datasets

def preprocess_GarageFinish(*datasets):
    for dataset in datasets:
        dataset['GarageFinish'].fillna(value='None', inplace=True)
    return datasets

def preprocess_GarageQual(*datasets):
    for dataset in datasets:
        dataset['GarageQual'].fillna(value='None', inplace=True)

    return datasets

def preprocess_GarageCond(*datasets):
    for dataset in datasets:
        dataset['GarageCond'].fillna(value='None', inplace=True)
    return datasets

def preprocess_PoolQC(*datasets):
    for dataset in datasets:
        dataset['PoolQC'].fillna(value='None', inplace=True)
    return datasets

def preprocess_Fence(*datasets):
    for dataset in datasets:
        dataset['Fence'].fillna(value='None', inplace=True)
    return datasets

def preprocess_MiscFeature(*datasets):
    for dataset in datasets:
        dataset['MiscFeature'].fillna(value='None', inplace=True)
    return datasets

def preprocess_Utilities(*datasets):
    for dataset in datasets:
        dataset['Utilities'].fillna(value='None', inplace=True)
    return datasets

def preprocess_Exterior1st(*datasets):
    for dataset in datasets:
        dataset['Exterior1st'].fillna(value='None', inplace=True)
    return datasets

def preprocess_Exterior2nd(*datasets):
    for dataset in datasets:
        dataset['Exterior2nd'].fillna(value='None', inplace=True)
    return datasets

def preprocess_KitchenQual(*datasets):
    for dataset in datasets:
        dataset['KitchenQual'].fillna(value='None', inplace=True)
    return datasets

def preprocess_Functional(*datasets):
    for dataset in datasets:
        dataset['Functional'].fillna(value='None', inplace=True)
    return datasets

def preprocess_SaleType(*datasets):
    for dataset in datasets:
        dataset['SaleType'].fillna(value='None', inplace=True)
    return datasets

def preprocess_GarageYrBlt(*datasets):
    GarageYrBlt_mean = datasets[0]['GarageYrBlt'].mean()
    for dataset in datasets:
        dataset['GarageYrBlt'].fillna(value=GarageYrBlt_mean, inplace=True)
    return datasets

def preprocess_MasVnrArea(*datasets):
    MasVnrArea_mean = datasets[0][datasets[0]['MasVnrType'] != 'None']['MasVnrArea'].mean()
    for dataset in datasets:
        dataset['MasVnrArea'].apply(lambda x: x if x is not None else MasVnrArea_mean)
    return datasets

def preprocess_MoSold(*datasets):
    for dataset in datasets:
        dataset['MoSold_sin'] = np.sin(dataset['MoSold']-1 * (2. * np.pi / 12))
        dataset['MoSold_cos'] = np.cos(dataset['MoSold']-1 * (2. * np.pi / 12))
        dataset.drop(columns=['MoSold'], inplace=True)
    return datasets