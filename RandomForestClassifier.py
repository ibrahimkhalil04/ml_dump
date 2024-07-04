
from collections import Counter
import numpy as np 
from itertools import combinations_with_replacement
import DecisionTreeClassifier # type: ignore
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Prep the data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

class RandomForest:

    def __init__(self, n_trees = 25, n_jobs=2 ) -> None:
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        num_features = int(np.sqrt(X.shape[1]))
        for _ in range(self.n_trees):
            samples_indices = np.random.choice(X.shape[0], size=n_samples, replace=True)
            bootstrap_samples = X[samples_indices, :]
            y_samples = y[samples_indices]
            dTree = DecisionTreeClassifier.DecisionTreeClassifier(max_depth=4, tree_type='random_forest', num_features=num_features)
            dTree.fit(bootstrap_samples, y_samples)
            self.trees.append(dTree)
        

    def predict(self, X):
        all_predictions = []
        for tree in self.trees:
            prediction = tree.predict(X)
            all_predictions.append(prediction)
        # Transpose to get predictions for each data point
        all_predictions = np.array(all_predictions).T # Shape: (num_data_points, num_trees)
        # Aggregate predictions by majority vote 
        final_predictions = []
        for preds in all_predictions:
            majority_vote = Counter(preds).most_common(1)[0][0]
            final_predictions.append(majority_vote)
        return final_predictions
        


    def accuracy_score(self, y_pred: np.array, y_true: np.array):
        return np.sum(y_pred == y_true) / y_true.size
    


randomForest = RandomForest(n_trees=25, n_jobs=2)
randomForest.fit(X_train, y_train)
y_preds = randomForest.predict(X_test)
score = randomForest.accuracy_score(y_preds, y_test)
print("Accuracy score: ", score)


forest = RandomForestClassifier(n_estimators=25, n_jobs=2, random_state=1)
forest.fit(X_train, y_train)
y_forest = forest.predict(X_test)
score_forest = accuracy_score(y_test, y_forest )
print("Forest score : ", score_forest)
