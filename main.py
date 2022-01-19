from helper_func import load_data, data_preprosessing, train_model_rf,\
    data_prep_predict, predict_default

CORR_THRESHOLD = 0.8
TARGET = "default"
SELECTED_FEATURES = []
categorical_features = ["merchant_category", "merchant_group", "name_in_email"]


if __name__ == "__main__":
    # load config file #todo

    # Load Data
    df_train, df_predict = load_data()
    print(df_train)

    # Data preprosessing
    X, y = data_preprosessing(df_train, CORR_THRESHOLD)
    # Train model
    train_model_rf(X, y)
    # Prediction
    X_prep = data_prep_predict(
        data, target, selected_features, categorical_features)
    # predict_default(df_predict)
