# Loan default predictor
### Case study problem definition
- Predict the probability that a given user will default their next payment.
### Purpose 
Implementation of a supervised learning model to predict the probability of default.

### Data
- In /data/dataset.csv

### Train model

To train the model and predict the probability of default for the data points in dataset where "default" values are missing, first install the requirements and then execute main.py script:
```
python main.py
```
### API
Model is exposed through api using flask.
To start the app, we run:
```
python app.py
```
### Docker 
To setup application in docker,we first need to train the model and then we run:
```
docker-compose build 
docker-compose up -d
```
Default port on which application is exposed is 5000




