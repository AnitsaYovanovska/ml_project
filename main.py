import json
import sys
from helper_func import load_data, data_preprosessing, train_model_rf,\
    data_prep_predict, predict_default, json_api_to_dataframe, predict_default_api,\
    load_config_file
import json


if __name__ == "__main__":

    # load config file
    target, selected_features, categorical_features, corr_threshold = load_config_file()

    # 1.Load Data
    df_train, df_predict = load_data()
    # print(df_train)

    # use part  2 and 3 when training the model
    # 2.Data preprosessing
    X, y = data_preprosessing(df_train, corr_threshold)
    # 3.Train model
    train_model_rf(X, y)

    # use part 4 to make prediction for df_predict
    # 4.Prediction
    X_prep = data_prep_predict(df_predict, target, selected_features, categorical_features)
    predict_default(X_prep, "all")


