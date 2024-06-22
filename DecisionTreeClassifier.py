
import numpy as np 


class Node:
    def __init__(self, features=None, target=None, threshold=None, left=None, right=None, split_index=-1): # change the split_inx
        self.features = features
        self.target = target
        self.threshold = threshold
        self.left = left
        self.right = right
        self.split_index = split_index
    
    

    def split_node(self):
        if self.features == None or self.threshold == None or self.split_index == max:
            return
        lefts = self.features[:, self.split_index] <= self.threshold
        left_features = self.features[lefts]
        right_features = self.features[~lefts]
        left_target = self.target[lefts]
        right_target = self.target[~lefts]
        self.left = Node(features=left_features, target=left_target)
        self.right = Node(features=right_features, target=right_target)
        

    def gini_impurity(self, y: np.array):
        # Compute the Gini impurity for a list of labels y
        uniq, counts = np.unique(y, return_counts=True)
        p = counts / y.size
        gini = 1 - np.sum(p**2)
        return gini


    def information_gain(self, criterion):
        # Compute the information gain from splitting y into y_left and y_right
        self.split_node()
        if criterion == 'gini':
            impurity = self.gini_impurity(self.target)
            impurity_left = self.gini_impurity(self.left.target)
            impurity_right = self.gini_impurity(self.right.target)
        elif criterion == 'entropy':
            pass # Will implement later 
        elif criterion == 'error':
            pass # Will implement later 
        else:
            raise ValueError("Invalid criterion. Supported criteria are 'gini', 'entropy', and 'error'.")
        weight_left = self.left.target.size / self.target.size
        weight_right = self.right.target.size / self.target.size
        return impurity - (weight_left*impurity_left + weight_right*impurity_right)
    
    def is_pure(self, criterion):
        if self.information_gain(criterion) <= 0.0:
            return True
        if np.unique(self.target).size <= 1:
            return True
        else :
            return False




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
