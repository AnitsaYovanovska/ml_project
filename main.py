import json
import sys
from helper_func import load_data, data_preprosessing, train_model_rf,\
    data_prep_predict, predict_default


if __name__ == "__main__":
    # load config file
    try:
        with open("model_config.json") as json_file:
            model_config = json.load(json_file)
    except FileNotFoundError as f_error:
        print(f_error)
        sys.exit(f"Can't open file : model_config.json")

    target = model_config["target"]
    selected_features = model_config["selected_features"]
    categorical_features = model_config["categorical_features"]
    corr_threshold = model_config["corr_threshold"]

    # 1.Load Data
    df_train, df_predict = load_data()
    # print(df_train)

    # use part  2 and 3 when training the model
    # # 2.Data preprosessing
    # X, y = data_preprosessing(df_train, corr_threshold)
    # # 3.Train model
    # train_model_rf(X, y)

    # 4.Prediction
    X_prep = data_prep_predict(df_predict, target, selected_features, categorical_features)
    predict_default(X_prep)
