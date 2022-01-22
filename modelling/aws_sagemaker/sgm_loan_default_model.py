import json
import sys
import os
import json
# import logging
import argparse
# from sklearn.externals import joblib
# import joblib
import pandas as pd
import numpy as np
# from collinearity import SelectNonCollinear
from sklearn.preprocessing import OneHotEncoder
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,
from sklearn.ensemble import RandomForestClassifier
import pickle



CORR_THRESHOLD = 0.8
TARGET = "default"
SELECTED_FEATURES = [
        "account_amount_added_12_24m", 
        "account_days_in_dc_12_24m",
        "account_days_in_rem_12_24m",
        "account_days_in_term_12_24m",
        "age",
        "avg_payment_span_0_12m",
        "merchant_category",
        "merchant_group",
        "has_paid",
        "max_paid_inv_0_12m",
        "name_in_email",
        "num_active_div_by_paid_inv_0_12m",
        "num_active_inv",
        "num_arch_dc_0_12m",
        "num_arch_dc_12_24m",
        "num_arch_ok_12_24m",
        "num_arch_rem_0_12m",
        "num_arch_written_off_0_12m",
        "num_arch_written_off_12_24m",
        "num_unpaid_bills",
        "status_last_archived_0_24m",
        "status_2nd_last_archived_0_24m",
        "status_3rd_last_archived_0_24m",
        "status_max_archived_0_6_months",
        "status_max_archived_0_12_months",
        "recovery_debt",
        "sum_capital_paid_account_0_12m",
        "sum_capital_paid_account_12_24m",
        "sum_paid_inv_0_12m",
        "time_hours"
    ]
CATEGORICAL_FEATURES = [
    "merchant_category",
    "merchant_group",
    "name_in_email"]


def load_data(args):
    """Load data from csv"""

    file = os.path.join(args.train, "dataset.csv")
    df_data = pd.read_csv(file, sep=';', engine="python")

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

    # np.savetxt(os.path.join("modelling", "removed_col_50_pct.csv"),
    #            col_to_delete,
    #            delimiter=",",
    #            fmt='% s')

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

#     selector = SelectNonCollinear(correlation_threshold=corr_threshold)

#     features = X.select_dtypes(include=["number"]).columns.to_list()

#     X_arr = X.select_dtypes(include=["number"]).to_numpy()

#     selector.fit(X_arr, np.ravel(y))
#     mask = selector.get_support()

#     remove_corr_col = list(
#         set(X.select_dtypes(include=["number"]))-set(np.array(features)[mask]))

    remove_corr_col = ["num_arch_ok_0_12m", "status_max_archived_0_24_months", "max_paid_inv_0_24m"]

    # remove the highly correlated columns
    X.drop(remove_corr_col, axis=1, inplace=True)

    # np.savetxt(os.path.join("modelling", "removed_col_collinearity.csv"),
    #            remove_corr_col,
    #            delimiter=",",
    #            fmt='% s')

    print("Removed highly correlated columns:{}".format(remove_corr_col))

    return X, y


def data_preprosessing_onehot(X):
    """returns dataframe with Onehot encoded categorical variables"""

    categorical_col = X.select_dtypes(include=["object"]).columns.to_list()
    X_categorical = X[categorical_col]

    encoder = OneHotEncoder(drop='first')
    encoder.fit(X_categorical)
    X_onehot = encoder.transform(X_categorical)

    X_onehot_df = pd.DataFrame(X_onehot.toarray())

    # pickle.dump(encoder, open(os.path.join(
    #     "modelling", "one_hot_encoder.pkl"), 'wb'))

    X_last = pd.concat([X.drop(["merchant_category", "merchant_group",
                       "name_in_email"], axis=1), X_onehot_df.set_index(X.index)], axis=1)

    return X_last


def data_preprosessing_oversampling(X, y):
    """returns oversampled dataset and balanced target feature labels"""

    # transform the dataset
    # oversample = SMOTE(random_state=101)
    # X, y = oversample.fit_resample(X, y)

    return X, y


def train_model_rf(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.3, stratify=y)

    # train a randomforest classifier
    rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    rf_clf.fit(X_train, np.ravel(y_train))

    # save trained model
    pickle.dump(rf_clf, open(os.path.join(
        args.model_dir, "rf_clf_model.pkl"), 'wb'))


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
    # X, y = data_preprosessing_oversampling(X, y)

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    # LOAD DATA 
    df_train, df_predict = load_data(args)

    # TRANSFORM
    X, y = data_preprosessing(df_train, CORR_THRESHOLD)

    # TRAIN MODEL
    train_model_rf(X, y, args)


def model_fn(model_dir):
    """Load fitted model"""

    rf_clf = pickle.load(open(os.path.join(model_dir, "rf_clf_model.pkl"), 'rb'))
    return rf_clf