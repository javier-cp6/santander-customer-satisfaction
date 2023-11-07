# Predicting customer satisfaction using Classification
**Machine Learning Zoomcamp - Midterm Project**  
Binary classification project for the Santander Customer Satisfaction competition hosted on Kaggle.  


## 1. Problem description

Customer satisfaction is a fundamental aspect of any business. It is a critical metric for assessing the success of a company's products or services, as happy customers are more likely to remain loyal and contribute positively to a company's growth. 

In this project, the [Santander Customer Satisfaction Kaggle competition](https://www.kaggle.com/competitions/santander-customer-satisfaction)  is leveraged to develop and deploy a machine learning model that can predict unsatisfied customers in the early stages. The results will allow the company to take proactive actions to improve customer relationships.


## 2. Data

The Santander Customer Satisfaction competition provides an anonymized dataset containing 370 features and the "TARGET" column which is the variable to predict. It equals 1 for unsatisfied customers and 0 for satisfied customers.

The task is to predict the probability that each customer in the test set is an unsatisfied customer.

File descriptions:
* `train.csv` - the training set including the target
* `test.csv` - the test set without the target
* `sample_submission.csv` - a sample submission file in the correct format

The files in the dataset can be downloaded using a Kaggle API Key or directly in the [competition web page](https://www.kaggle.com/competitions/santander-customer-satisfaction/data).


## 3. Deployment

The `notebook` in this project describes the Machine Learning process including the Exploratory Data Analyis, data cleaing and preparation, training, tuning, and deployment of the final model.

Scripts:  
* `train.py` - script for training the final XGBoost model. It saves the trained model as `model_xgb.bin`.
* `predict.py` - script for serving the model.
* `predict-test.py`- script to make a test prediction.


### 3.1 Environment
Install packages from existing Pipfile and Pipfile.lock files.

Run the following command (it requires Python 3.9):
```bash
$ pipenv install
```

### 3.2 Serving model with Flask

Run the Flask app `predict.py` on the pipenv environment:

```bash
$ pipvenv shell
$ python predict.py
```
Also works with:
```bash
$ pipenv run python predict.py
```

Make a request to the app:
```bash
python predict-test.py
```

It will return the following response: `{'probability': 0.07192421704530716, 'unsatisfied': True}`

### 3.3 Serving model with Docker

Making request using a Docker container:

Build a Docker image:
```bash
$ docker build -t santander-customer-satisfaction .
```

Run a Docker container:
```bash
$ docker run -it --rm -p 9696:9696 santander-customer-satisfaction:latest
```

Make an inference using the model served by the container by running the following script
```
$ python predict-test.py
```

It will return the following response: `{'probability': 0.07192421704530716, 'unsatisfied': True}`


## 4. Results
The final XGBoost model achieved an AUC score of 0.8394 surpassing the scores obtained by the decision tree and random forest models. 

## 5. Deliverables
* `README.md`
* Data: The files in the dataset can be downloaded using a Kaggle API Key or directly in the [competition web page](https://www.kaggle.com/competitions/santander-customer-satisfaction/data).
* `notebook.ipynb`)
* `train.py`
* `predict.py`
* `predict-test.py`
* `Pipenv` and `Pipenv.lock`
* `Dockerfile`


## Sources: 
Soraya_Jimenez, Will Cukierski. (2016). Santander Customer Satisfaction. Kaggle. https://kaggle.com/competitions/santander-customer-satisfaction
