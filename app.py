import json
import sys
from helper_func import  data_prep_predict, predict_default, json_api_to_dataframe, load_config_file
import os
import json
import logging
# import flask
from flask import Flask, request

# Define logging LEVEL, output type
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s')


# Flask app definition
app = Flask("predictor")


# /predict
@app.route('/predict', methods=['GET'])
def post():
    content = request.get_json()
    print(content)
    try:
        target, selected_features, categorical_features, corr_threshold = load_config_file()
        df_to_predict = json_api_to_dataframe(content)
        # data preprosessing
        X_prep = data_prep_predict(df_to_predict, target, selected_features, categorical_features)

        # prediction
        prediction = predict_default(X_prep, 'api')

        return json.dumps(prediction), 200
    except:
        logging.error("Can't make prediction for provided data {}".format(content))
        return json.dumps({"error": "Can't make prediction for provided data {}".format(content)}), 404


if __name__ == "__main__":

    app.run(host='0.0.0.0', debug=True, port=5000)

