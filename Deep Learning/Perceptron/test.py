import Perceptron as p
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)
print(df.tail()) #To see the last lines

y_train = df.iloc[0:10, 4].values
y_train = np.where(y_train == 'Iris-setosa', -1, 1)

X_train = df.iloc[0:10, :4].values

ppn = p.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X_train, y_train)

