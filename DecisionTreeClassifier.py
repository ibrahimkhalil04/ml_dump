
import numpy as np 


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_impurity(y: np.array):
    # Compute the Gini impurity for a list of labels y
    uniq, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    gini = 1 - np.sum(p**2)
    return gini


def information_gain(y, y_left, y_right, criterion):
    # Compute the information gain from splitting y into y_left and y_right
    if criterion == 'gini':
        impurity = gini_impurity(y)
        impurity_left = gini_impurity(y_left)
        impurity_right = gini_impurity(y_right)
    elif criterion == 'entropy':
        pass # Will implement later 
    elif criterion == 'error':
        pass # Will implement later 
    else:
        raise ValueError("Invalid criterion. Supported criteria are 'gini', 'entropy', and 'error'.")
    weight_left = y_left.size / y.size
    weight_right = y_right.size / y.size
    return impurity - (weight_left*impurity_left + weight_right*impurity_right)




def best_split(X, y, criterion):
    # Find the best feature and threshold to split the data
    pass

def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, criterion='gini'):
    # Recursively build the tree
    pass

def stopping_criteria(depth, max_depth, n_samples, min_samples_split):
    # Determine if we should stop splitting
    pass

def predict_tree(node, X):
    # Traverse the tree and make predictions for X
    pass

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = build_tree(X, y, 0, self.max_depth, self.min_samples_split, self.criterion)

    def predict(self, X):
        return [predict_tree(self.root, x) for x in X]
