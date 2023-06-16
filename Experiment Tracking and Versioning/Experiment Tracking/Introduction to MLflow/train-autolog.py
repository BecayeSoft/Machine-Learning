import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle


# MLflow
import mlflow

mlflow.set_experiment("Iris Classification") 

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()


# Load the data
data = pd.read_csv('../../data/iris/v1/iris.data', header=None)
data = shuffle(data, random_state=42)
features = data.iloc[:, :-1]
target = data.iloc[:, -1]


# Shuffle data & split into training and testing sets
features, target = shuffle(features, target, random_state=42)
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)


# Train a Random Forest classifier
clf = RandomForestClassifier(
        n_estimators=7,
        min_samples_split=2,
        max_depth=5,
        criterion="entropy",
        random_state=42
)

clf.fit(X_train, y_train)


# Make predictions on the test set
y_pred = clf.predict(X_test)            
