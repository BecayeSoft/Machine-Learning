# Classification
CLassificaiton is a type of supervised learning used to predict the class of data. 
It can be used for instance for image recognition, spam filtering, and fraud detection.

The notebooks assume that your target (the variable you are trying to predict) is located at the last column.
Therefore, we separate the features from the target like this:
```
dataset = pd.read_csv('Data.csv')
features = dataset.iloc[:, :-1].values         # get all column except the last one (target)
target = dataset.iloc[:, -1].values            # get the last column (target)
```

After choosing a dataset to predict a class, you may use any of the following algorithms:

* Decision Tree Classification
* K Nearest Neighbors
* Kernel SVM
* Logistic Regression
* Naive Bayes
* Random Forest Classification
* Support Vector Machine
