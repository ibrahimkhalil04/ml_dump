"""
Decision Tree Classifier Module

This module implements a Decision Tree Classifier from scratch. 
It includes the necessary components to build, train, and evaluate a decision tree model.

Classes
-------
Node:
    Represents a node in the decision tree. Handles splitting of data and 
    calculation of impurities and information gain.

DecisionTreeClassifier:
    Implements the decision tree classifier. Provides methods for training the model (`fit`), 
    making predictions (`predict`), and evaluating model performance (`accuracy_score`).

Functions
---------
best_split(node: Node, criterion: str) -> tuple:
    Finds the best feature and threshold to split the data at the given node.

build_tree(node: Node, depth: int = 0, max_depth: int = None, min_samples_split: int = 2, criterion: str = 'gini') -> Node:
    Recursively builds the decision tree.

stopping_criteria(node: Node, depth: int, max_depth: int, n_samples: int, min_samples_split: int, criterion: str = 'gini') -> bool:
    Determines if the current node should stop splitting.

predict_tree(node: Node, X: np.ndarray) -> int:
    Traverses the tree and makes a prediction for a single sample.
"""

import random
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier as DTClassifier
from sklearn import tree
import math




class Node:
    """
    A class representing a node in a decision tree.

    Attributes:
    ----------
    features : numpy.ndarray
        The input features for the node.
    target : numpy.ndarray
        The target labels for the node.
    threshold : float
        The threshold value for splitting the node.
    left : Node or None
        The left child node.
    right : Node or None
        The right child node.
    split_index : int
        The index of the feature used for splitting the node.
    info_gain : float
        The information gain achieved by splitting the node.

    Methods:
    --------
    split_node():
        Splits the current node into left and right child nodes based on the threshold.
    gini_impurity(y: np.array):
        Computes the Gini impurity for the given target labels.
    information_gain(criterion):
        Computes the information gain from splitting the node using the specified criterion.
    is_pure():
        Checks if the node is pure (homogeneous).
    """
    def __init__(self, features=None, target=None, depth=0, threshold=0.0, left=None, right=None, split_index=-1, iG = 0.0): # change the split_inx
        self.features = features
        self.target = target
        self.threshold = threshold
        self.depth = depth
        self.left = left
        self.right = right
        self.split_index = split_index 
        self.info_gain = iG
    
    def split_node(self):
        """
        Splits the node into left and right child nodes based on the current threshold.

        Returns:
        --------
        tuple
            A tuple containing the left and right child nodes.
        """
        if self.features is None or self.threshold == 0.0 or self.split_index == -1:
            return
        lefts = self.features[:, self.split_index] <= self.threshold
        left_features = self.features[lefts]
        right_features = self.features[~lefts]
        left_target = self.target[lefts]
        right_target = self.target[~lefts]
        return (Node(features=left_features, target=left_target, depth=self.depth+1), Node(features=right_features, target=right_target, depth=self.depth+1))

    def gini_impurity(self, y: np.array):
        """
        Computes the Gini impurity for the given target labels.

        Parameters:
        -----------
        y : numpy.ndarray
            The target labels.

        Returns:
        --------
        float
            The Gini impurity value.
        """
        uniq, counts = np.unique(y, return_counts=True)
        p = counts / y.size
        gini = 1 - np.sum(p**2)
        return gini

    def information_gain(self, criterion):
        # Compute the information gain from splitting y into y_left and y_right
        left_node, right_node = self.split_node()
        if criterion == 'gini':
            impurity = self.gini_impurity(self.target)
            impurity_left = self.gini_impurity(left_node.target) if left_node.target.size > 0 else 0
            impurity_right = self.gini_impurity(right_node.target) if right_node.target.size > 0 else 0
        elif criterion == 'entropy':
            pass # Will implement later 
        elif criterion == 'error':
            pass # Will implement later 
        else:
            raise ValueError("Invalid criterion. Supported criteria are 'gini', 'entropy', and 'error'.")
        weight_left = left_node.target.size / self.target.size if left_node.target.size > 0 else 0
        weight_right = right_node.target.size / self.target.size if right_node.target.size > 0 else 0
        return impurity - (weight_left*impurity_left + weight_right*impurity_right)
    
    def is_pure(self):
        if self.info_gain <= 0.01: # min info gain
            return True
        if np.unique(self.target).size == 1:
            return True
        else :
            return False



def best_split(node: Node, criterion, tree_type='simple', num_features=None):
    """
    Finds the best feature index and threshold to split the data based on the specified criterion.

    Parameters:
    -----------
    node : Node
        The current node in the decision tree where the split is to be evaluated.
    criterion : str
        The criterion to use for evaluating the split ('gini', 'entropy', 'error').
    tree_type : str, optional
        The type of tree ('simple' or 'random_forest'). Default is 'simple'.
    num_features : int, optional
        The number of features to consider for splitting when tree_type is 'random_forest'. 
        Default is None.

    Returns:
    --------
    tuple
        A tuple containing the best feature index, best threshold, and corresponding score (information gain).

    Notes:
    ------
    This function iterates over each feature and its unique thresholds to evaluate the information gain
    achieved by splitting the node. It returns the feature index, threshold, and information gain score
    that produce the highest gain according to the specified criterion.
    """
    if tree_type == 'random_forest':
        if num_features is None:
            num_features = int(math.sqrt(node.features.shape[1]))
        selected_features_indices = random.sample(range(node.features.shape[1]), k=num_features)
        selected_features = node.features[:, selected_features_indices]
    else :
        selected_features = node.features
        selected_features_indices = range(selected_features.shape[1])

    best_split_indx = -1
    best_score = 0.0
    best_threshold = 0.0

    for i, idx  in enumerate(selected_features_indices):
        thresholds = np.unique(selected_features[:, i])
        for threshold in thresholds:
            node.split_index = idx
            node.threshold = threshold
            iG = node.information_gain(criterion=criterion)
            if iG > best_score:
                best_score = iG
                best_split_indx = idx
                best_threshold = threshold
    # Out of the loop, best_split_indx, thereshold and score correspond to the best feature index, threshold to split the node 
    # and the information gain of the node
    return (best_split_indx, best_threshold, best_score)


def build_tree(node: Node, max_depth=None, min_samples_split=2, criterion='gini', tree_type='simple', num_features=None):
    """
    Recursively builds the decision tree by splitting nodes based on the best feature and threshold.

    Parameters:
    -----------
    node : Node
        The current node in the decision tree to be split.
    depth : int, optional (default=0)
        The current depth of the node in the tree.
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, grow the tree until all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    criterion : str, optional (default='gini')
        The function to measure the quality of a split ('gini', 'entropy', or 'error').
    tree_type : str, optional
        The type of tree ('simple' or 'random_forest'). Default is 'simple'.
    num_features : int, optional
        The number of features to consider for splitting when tree_type is 'random_forest'. Default is None.

    Returns:
    --------
    Node
        The root node of the constructed decision tree.

    Notes:
    ------
    This function recursively splits nodes by finding the best feature index and threshold to maximize information gain
    based on the specified criterion. It stops recursion when the stopping criteria (pure node, max depth, or insufficient samples)
    are met.
    """

    split_indx, threshold, score =  best_split(node, criterion, tree_type, num_features)
    if split_indx != -1 and threshold != 0.0:
        node.split_index = split_indx
        node.threshold = threshold
        node.info_gain = score
    if stopping_criteria(node, max_depth, node.target.size, min_samples_split):
        return node
    
    node.left, node.right = node.split_node()
    # Recursively build the tree for left and right child nodes
    node.left = build_tree(node.left, max_depth, min_samples_split, criterion)
    node.right = build_tree(node.right, max_depth, min_samples_split, criterion)
    return node



def stopping_criteria(node : Node, max_depth, n_samples, min_samples_split):
    # Determine if we shuld stop splitting
    if node.is_pure() :
        return True
    if node.depth >= max_depth: 
        return True
    if n_samples <= min_samples_split:
        return True
    return False

def predict_tree(node: Node, X, max_depth):
    # Traverse the tree and make predictions for X
    if node is None:
        return
    if node.is_pure() or node.depth >= max_depth :
        uniq, counts = np.unique(node.target, return_counts=True)
        return uniq[counts.argmax()]
    if X[node.split_index] <= node.threshold:
        return predict_tree(node.left, X, max_depth)
    else:
        return predict_tree(node.right, X, max_depth)


class DecisionTreeClassifier:
    ''' Class representing the decision tree classifier.

    Attributes
    ----------
    max_depth : int or None, optional (default=Node)
        the maximun depth of the tree .
    min_samples : int, optional (default=2)
        the minimun number of samples that is required to split a node in the tree.
    criterion : str, optional (default='gini')
        The function to measure the quality of a split ('gini', 'entropy', or 'error'). 
    root : Node
        The root node of the tree.
    
    Methods
    -------
    fit(X, y):
        trains the decision tree classifier on the provided features (X) and  target (y)
    predict(X):
        Predicts the target values for the provided features (X) using the trained decision tree.
    accuracy_score(y_pred, y_true):
        computes the accuracy score of the model based on predicted (y_pred) and true (y_true) target values.
    
    '''
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini', tree_type='simple', num_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.tree_type = tree_type
        self.num_features = num_features

    def fit(self, X, y):
        node = Node(X, y)
        self.root = build_tree(node, self.max_depth, self.min_samples_split, self.criterion, self.tree_type, self.num_features)

    def predict(self, X):
        return np.array([predict_tree(self.root, x, self.max_depth) for x in X])
    
    def accuracy_score(self, y_pred: np.array, y_true: np.array):
        return np.sum(y_pred == y_true) / y_true.size


# 
def print_tree(node: Node, d):
    print('Depth: ', d)
    print('\n')
    print([t for t in node.target])
    if node.left is not None:
        print_tree(node.left, d+1)
    if node.right is not None:
        print_tree(node.right, d+1)
    return






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
        


def main():

    # Prep the data
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    tree_model = DecisionTreeClassifier(max_depth=4)
    tree_model.fit(X_train, y_train)
    predictions =  tree_model.predict(X_test)
    print('Accuracy score : %.3f' % tree_model.accuracy_score(np.array(predictions), y_test))

    X_combined = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined, y=y_combined, classifier=tree_model, test_idx=range(105, 150))
    plt.xlabel("Petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.title('Trained with my own implementation of a decision tree model')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    sk_tree_model = DTClassifier(criterion='gini', max_depth=4, random_state=1)
    sk_tree_model.fit(X_train, y_train)
    X_combined = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined, y=y_combined, classifier=sk_tree_model, test_idx=range(105, 150))
    plt.xlabel("Petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.title('Trained with sklearn\'s decision tree model')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()