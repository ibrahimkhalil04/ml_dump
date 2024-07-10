"""


"""

from collections import Counter
import numpy as np
import matplotlib.pylab as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
import time 
 

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
            distances = self.calculate_distance(x)
            nearest_neighbors = self.find_knn(distances)
            majority_label = max(set(nearest_neighbors), key = list(nearest_neighbors).count)
            predictions.append(majority_label)
            
        return np.array(predictions)
    

    def accuracy_score(self, y_pred: np.array, y_true: np.array):
        return np.sum(y_pred == y_true) / y_true.size
    
    def calculate_distance(self, xi):
        if self.metric == 'minkowski':
            return np.sum(np.abs(xi - self.data['feature']) ** self.p, axis=1) ** (1 / self.p)
        else:
            raise ValueError('Unsupported metric: ' + self.metric)

    def find_knn(self, distances):
        nearest_indices = np.argsort(distances)[: self.n_neighbors]
        return y_train[nearest_indices]




class KNeighbors_with_KDTree:

    def __init__(self, n_neighbors) -> None:
        self.n_neighbors =n_neighbors
        self.tree = None

    def fit(self, X, y):
        self.tree = Kd_tree()
        self.tree.fit(X, y)

    def predict(self, X):
        y_preds = []
        for x in X:
            neighbors = self.tree.nearest_neighbors(x, self.n_neighbors)
            majority_vote = Counter(neighbors).most_common(1)[0][0]
            y_preds.append(majority_vote)
        return np.array(y_preds)
    
    def accuracy_score(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / y_true.size



class KdNode:
    '''' class representing a node in the kd tree 
    Attributes:
    ----------
    data: numpy array
        the data stored in the node
    left: KdNode
        the left child node 
    right: KdNode
        the right child node 
    depth: int
        the depth of the node
    cd: int
        the cutting dimension, the dimension used to split the points into left and right 
    '''

    def __init__(self, features, target, left=None, right=None, depth=0, cd=0) -> None:
        self.features = features
        self.target = target
        self.left = left
        self.right = right
        self.depth = depth
        self.cd = cd

    def get_elem():
        pass


    def isleaf(self):
            return True if self.left is None and self.right is None else False




class Kd_tree:

    def __init__(self) -> None:
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        if len(X) == 0:
            return None
        cd = depth % X.shape[1]
        sorted_indices = X[:, cd].argsort()
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        median_index = len(X_sorted) // 2
        median_features = X_sorted[median_index]
        median_target = y_sorted[median_index]

        node = KdNode(features=median_features, target=median_target, depth=depth, cd=cd)
        node.left = self.build_tree(X[:median_index], y[: median_index], depth=depth+1)
        node.right = self.build_tree(X[median_index+1:], y[median_index+1:], depth=depth+1)
        return node
        
   

    
    def query(self, node, x, n_neighbors):
        if node.isleaf():
            distance = np.sqrt(np.sum((x - node.features) ** 2))
            return [(node.features, node.target,  distance)]
        neighbors = []
        other_branch = None
        # Decide which branch to explore first
        if x[node.cd] < node.features[node.cd] and node.left is not None:
            other_branch = node.right
            neighbors.extend( self.query(node.left, x, n_neighbors))
        elif x[node.cd] >= node.features[node.cd] and node.right is not None:
            other_branch = node.left
            neighbors.extend( self.query(node.right, x, n_neighbors))            
        # If current is None, calculate the distance for the current node
        if not other_branch :
            distance = np.sqrt(np.sum((x - node.features) ** 2))
            return [(node.features, node.target, distance)]
        else:
            len_neighbors = len(neighbors)
            neighbors = np.array(neighbors)
            neighbors = neighbors[neighbors[:, 2].argsort()]
            current_max_distance = neighbors[len_neighbors-1][2]
            dist_with_splitting_plane = np.abs(x[node.cd] - node.features[node.cd])
            if len_neighbors < n_neighbors or dist_with_splitting_plane <= current_max_distance:
                other_result = self.query(other_branch, x, n_neighbors)
                current_max_distance = other_result[0][2]
                neighbors = list(neighbors)
                neighbors.extend(other_result)

            if  len(neighbors) >= n_neighbors:
                neighbors = np.array(neighbors)
                neighbors = neighbors[np.argsort(neighbors[:, 2])[:n_neighbors]]
        

        return list(neighbors)
    


    def nearest_neighbors(self, x, n_neighbors):
        neighbors =  self.query(self.root, x, n_neighbors) 
        neighbors = np.array(neighbors)
        return neighbors[:, 1]


    def search(self, value, current: KdNode):
        if current is None:
            return None
        cd = current.cd
        if np.array_equal(current.features, value):
            return current
        if value[cd] < current.features[cd]:
            return self.search(current.left, value)
        else:
            return self.search(current.right, value)
        

    def insert(self, value, current=None, depth=0, cd=0):
        cd = depth % value.shape[0]
        if current is None:
            return KdNode(value, depth=depth, cd=cd)
        if current.features[cd] > value[cd]:
            current.left = self.insert(value, current.left, depth+1)
        else:
            current.right = self.insert(value, current.right, depth+1)
        return current

    



def test():
    tree = Kd_tree()
    tree.fit(X_train, y_train)
    p = tree.nearest_neighbors([3.5, 1.2], 5)
    #print(p)


 
test()

knn = KNeighbors_with_KDTree(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy score : %.3f' % knn.accuracy_score(np.array(y_pred), y_test))

knn_simple = KNeighbors(n_neighbors=5)
knn_simple.fit(X_train, y_train)

sklearn_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=2, metric='minkowski')
sklearn_knn.fit(X_train, y_train)
preds = sklearn_knn.predict(X_test)
print('Accuracy score from sklearn : %.3f' % accuracy_score(np.array(y_pred), y_test))






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
    start = time.process_time()
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    end = time.process_time()
    print('Time spent predicting the labels: %d ms' %((end -start)*10**3))
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
#plot_decision_regions(X=X_combined, y=y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel("Petal length [cm]")
plt.ylabel("petal width [cm]")
plt.title('Trained with my own implementation of K nearest neighbors model')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

plot_decision_regions(X=X_combined, y=y_combined, classifier=knn_simple, test_idx=range(105, 150))
