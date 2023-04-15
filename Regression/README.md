# Regression
Regression is a popular machine learning technique used for predicting continuous numerical values such as salaries or house prices.

The notebooks assume that your target (the variable you are trying to predict) is located at the last column.
Therefore, we separate the features from the target like this:
```
dataset = pd.read_csv('Data.csv')
features = dataset.iloc[:, :-1].values         # get all column except the last one (target)
target = dataset.iloc[:, -1].values            # get the last column (target)
```

After choosing a dataset to predict a continuous numerical value, you may use any of the following algorithms:

* Decision Tree Regression
* Multiple Linear Regression
* Polynomial Regression
* Random Forest Regression
* Simple Linear Regression
* Support Vector Regression
