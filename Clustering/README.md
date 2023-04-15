# CLustering
Clustering is an unsupervised machine learning technique that involves grouping similar data points together based on their features or characteristics. it is unsupervised beacause it does not require labeled data.
Clustering is can be used for customer segmentation, image segmentation, and anomaly detection.

The notebooks assume that your target (the variable you are trying to predict) is located at the last column.
Therefore, we separate the features from the target like this:
```
dataset = pd.read_csv('Data.csv')
features = dataset.iloc[:, :-1].values         # get all column except the last one (target)
target = dataset.iloc[:, -1].values            # get the last column (target)
```

After choosing a dataset to predict a class, you may use any of the following algorithms:

* K Means Clustering
* Hierarchical Clustering
