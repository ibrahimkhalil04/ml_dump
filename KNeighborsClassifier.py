"""


"""

import numpy as np
import matplotlib.pylab as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier 
 

#*********
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



class KNeighbors:
    """
    k-Nearest Neighbors classifier

    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        Number of neighbors to use.
    p : int, optional (default=2)
        Power parameter for the Minkowski metric.
    metric : str, optional (default='minkowski')
        The distance metric to use.
    """

    def __init__(self, n_neighbors = 5, p=2, metric='minkowski') -> None:
        self.n_neighbors = n_neighbors
        self.p = p
        self.metric = metric
        self.data = None

    def fit(self, X, y):
        self.data = {'feature': X,
                     'target': y}

    def predict(self, X):
        predictions = []
        for x in X:
            neighbors = find_neighbors(x, self.data['feature'], self.data['target'], self.n_neighbors, self.p, self.metric )
            neighbor_labels = [label for _, _, label in neighbors[1]]
            majority_label = max(set(neighbor_labels), key = neighbor_labels.count)
            predictions.append(majority_label)
        return np.array(predictions)
    
    def accuracy_score(self, y_pred: np.array, y_true: np.array):
        return np.sum(y_pred == y_true) / y_true.size



def find_neighbors(x, features=None, target=None, n_neighbors=5, p=2, metric='minkowski'):
    knn = find_knn(x, features, target, n_neighbors, p, metric)
    return (x, knn)

def calculate_distance(xi, xj, p, metric):
    if metric == 'minkowski':
        return np.sum(np.abs(xi - xj) ** p) ** (1 / p)
    else:
        raise ValueError('Unsupported metric: ' + metric)
        

def find_knn(xi=None, Xj=None, target=None, n_neighbors=5, p=2, metric = 'minkowski'):
    neighbors = []
    for i, xj in enumerate(Xj):
        distance = calculate_distance(xi, xj, p, metric)
        label = target[i]
        neighbors.append((distance, xj, label))
    nearest_neighbors = sorted(neighbors, key=lambda x : x[0])[:n_neighbors]
    return nearest_neighbors


knn = KNeighbors(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print('Accuracy score : %.3f' % knn.accuracy_score(np.array(y_pred), y_test))






#*********************************
def plot_decision_regions(X, y, classifier, resolution= 0.02, test_idx=None):
    #setup marker generator and color map
    markers = ['o', 's', '^', 'v', '<']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2, = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap= cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples 
    for idx, cl, in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1], 
                    alpha=0.8, 
                    c= colors[idx], 
                    marker= markers[idx], 
                    label= f'Class {cl}',
                    edgecolors='black')
    # Hightlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolors='black', 
                    alpha=1.0, linewidths=1, marker='o', s = 100, label= 'Test set')
        
        
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined, y=y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel("Petal length [cm]")
plt.ylabel("petal width [cm]")
plt.title('Trained with my own implementation of a decision tree model')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

