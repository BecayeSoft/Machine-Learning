# Tracking Iris data with MLflow

MLflow allows you to track your experiments by logging information such as hyperparameters, metrics or even models.

In this directory, there are two files that  train a Random Forest model on the iris dataset and manually log hyperparameters, metrics and the trained model itself:

* `train.py`: manually logs the information
* `train-autolog.py` automatically logs the information

<br>


<br>
<h3 style="color:rgb(136 19 55); font-weight:bold">Get Started</h3>
<br>

```
pip install -r requirements.txt         # or pip install mlflow
```


**Set experiment name**
```
mlflow.set_experiment("Iris Classification")
```

<br>
<h3 style="color:rgb(136 19 55); font-weight:bold">Auto Logging</h3>
<br>
## 
```
mlflow.sklearn.autolog()
```
The autolog functionality automatically captures and logs relevant information during the execution of your code, associating it with the active MLflow run.

[See the supported libraries here](https://mlflow.org/docs/latest/tracking.html#automatic-logging).


<br>
<h3 style="color:rgb(136 19 55); font-weight:bold">Manual Logging</h3>
<br>

You can also manually log experiments.

**Log hyperparameters**
```
params = {
    "n_estimators": 30,
    "min_samples_split": 2,
    "max_depth": 7,
    "criterion": "gini"
}

mlflow.log_params(params)
```

**Log metrics**
```
 metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro')
    }

    mlflow.log_metrics(metrics)
```

**Log model**
```
mlflow.sklearn.log_model(clf, "model")
```

<br>
<h3 style="color:rgb(136 19 55); font-weight:bold">Visualize experiments</h3>
<br>

```
mlflow ui
```
Then go to: http://127.0.0.1:5000/.
