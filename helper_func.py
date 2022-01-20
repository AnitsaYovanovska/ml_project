import os
import pandas as pd
import numpy as np
from collinearity import SelectNonCollinear
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
import pickle

# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report, roc_auc_score, accuracy_score, plot_roc_curve
# from sklearn.pipeline import Pipeline


def load_data():
    """Load data from csv"""

    path = os.path.join("data", "dataset.csv")
    df_data = pd.read_csv(path, sep=';')

    df_train = df_data[~df_data['default'].isna()]
    df_predict = df_data[df_data['default'].isna()]

    return df_train, df_predict


def data_preprosessing_missing(data):
    """returns dataframe without missing values"""

    # missing values
    percent_missing = round(data.isnull().sum()/len(data) * 100, 2)
    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                     'percent_missing': percent_missing})
    # delete columns where missing data > 50%
    col_to_delete = missing_value_df[missing_value_df['percent_missing'] > 49].index.to_list()

    np.savetxt(os.path.join("modelling", "removed_col_50_pct.csv"),
               col_to_delete,
               delimiter=",",
               fmt='% s')

    data = data.drop(col_to_delete, axis=1)
    print("Removed columns with missing data > 50%:{}".format(col_to_delete))

    missing_30_pct = missing_value_df[(missing_value_df['percent_missing'] > 0) & (
        missing_value_df['percent_missing'] < 30)].index.to_list()

    # delete entries where missing data < 30%
    # data = data.dropna()

    # imput median where missing data <30%
    for col in missing_30_pct:
        data[col].fillna((data[col].median()), inplace=True)

    return data


def data_preprosessing_collinearity(X, y, corr_threshold):
    """returns dataframe with eliminated correlated features"""

    selector = SelectNonCollinear(correlation_threshold=corr_threshold)

    features = X.select_dtypes(include=["number"]).columns.to_list()

    X_arr = X.select_dtypes(include=["number"]).to_numpy()

    selector.fit(X_arr, np.ravel(y))
    mask = selector.get_support()

    remove_corr_col = list(
        set(X.select_dtypes(include=["number"]))-set(np.array(features)[mask]))

    # remove the highly correlated columns
    X.drop(remove_corr_col, axis=1, inplace=True)

    np.savetxt(os.path.join("modelling", "removed_col_collinearity.csv"),
               remove_corr_col,
               delimiter=",",
               fmt='% s')

    print("Removed highly correlated columns:{}".format(remove_corr_col))

    return X, y


def data_preprosessing_scaler(X):

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    pickle.dump(scaler, open(os.path.join(
        "modelling", "standard_scaler.pkl"), 'wb'))

    return X_scaled


def data_preprosessing_onehot(X):
    """returns dataframe with Onehot encoded categorical variables"""

    categorical_col = X.select_dtypes(include=["object"]).columns.to_list()
    X_categorical = X[categorical_col]

    encoder = OneHotEncoder(drop='first')
    encoder.fit(X_categorical)
    X_onehot = encoder.transform(X_categorical)

    X_onehot_df = pd.DataFrame(X_onehot.toarray())

    pickle.dump(encoder, open(os.path.join(
        "modelling", "one_hot_encoder.pkl"), 'wb'))

    X_last = pd.concat([X.drop(["merchant_category", "merchant_group",
                       "name_in_email"], axis=1), X_onehot_df.set_index(X.index)], axis=1)

    return X_last


def data_preprosessing_oversampling(X, y):
    """returns oversampled dataset and balanced target feature labels"""

    # transform the dataset
    oversample = SMOTE(random_state=101)
    X, y = oversample.fit_resample(X, y)

    return X, y


def train_model_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.3, stratify=y)

    # train a randomforest classifier
    rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    rf_clf.fit(X_train, np.ravel(y_train))

    # save trained model
    pickle.dump(rf_clf, open(os.path.join(
        "modelling", "rf_clf_model.pkl"), 'wb'))


def data_preprosessing(data, corr_threshold):
    """prepares data for input to model, returns transformend dataframe"""

    # missing data
    data = data_preprosessing_missing(data)

    X = data.drop(["uuid", "default"], axis=1)
    y = data[["default"]]

    # remove collinearity
    X, y = data_preprosessing_collinearity(X, y, corr_threshold)
    # onehot encoder
    X = data_preprosessing_onehot(X)
    # oversampling
    X, y = data_preprosessing_oversampling(X, y)

    return X, y


def data_prep_predict(data, target, selected_features, categorical_features):
    """function to prepare data for input to model for prediction"""

    # check for default column in test data
    if target in data.columns:
        data = data[data["default"].isna()].drop(["default"], axis=1).reset_index(drop=True)

    # check for all required features
    for col_name in selected_features:
        if col_name not in data.columns:
            raise Exception('Required column  is missing:{}', format(col_name))

    print('Writing the data to csv file where required column values are missing')

    data[data[selected_features].isnull().any(axis=1)].to_csv(os.path.join(
        "predict", 'required_columns_values_missing.csv'))

    # data.dropna(subset=selected_features, inplace=True)

    # imput median where missing data <30%
    for col in selected_features:
        if data[col].isnull().sum() != 0:
            data[col] = data[col].fillna(data[col].median())
    
    selected_features.append('uuid')
    
    # filter selected features
    data = data[selected_features]

    # onehot encoding
    df_categorical = data[categorical_features]

    encoder = pickle.load(open(os.path.join("modelling", "one_hot_encoder.pkl"), 'rb'))
    data_one_hot = encoder.transform(df_categorical)
    df_data_one_hot = pd.DataFrame(data_one_hot.toarray())
    data_last = pd.concat([data.drop(categorical_features, axis = 1), 
                            df_data_one_hot.set_index(data.index)],axis=1)

    return data_last


def predict_default(data_last):
    """function to predict the probability of default and write the result to prediction_default.csv file"""

    # load trained model
    rf_clf = pickle.load(open(os.path.join("modelling", "rf_clf_model.pkl"), 'rb'))

    if 'uuid' in data_last.columns:

        data_last['pd_prediction'] = rf_clf.predict(data_last.drop(["uuid"], axis=1))
        df_predicted = data_last[['uuid', 'pd_prediction']]
    else:
        data_last['pd_prediction'] = rf_clf.predict(data_last)
        df_predicted = data_last['pd_prediction']

    df_predicted.to_csv(os.path.join("predict",'prediction_default.csv'))

    print('Check predicted pd in: ../predict/prediction_default.csv')

