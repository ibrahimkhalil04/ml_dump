
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# Prep the data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



class Node:
    def __init__(self, features=None, target=None, threshold=None, left=None, right=None, split_index=-1, iG = 0.0): # change the split_inx
        self.features = features
        self.target = target
        self.threshold = threshold
        self.left = left
        self.right = right
        self.split_index = split_index 
        self.info_gain = iG
    
    

    def split_node(self):
        if self.features is None or self.threshold is None or self.split_index == -1:
            return
        lefts = self.features[:, self.split_index] <= self.threshold
        left_features = self.features[lefts]
        right_features = self.features[~lefts]
        left_target = self.target[lefts]
        right_target = self.target[~lefts]
        return {'features': [left_features, right_features],
                'targets': [left_target, right_target]}

        #self.left = Node(features=left_features, target=left_target)
        #self.right = Node(features=right_features, target=right_target)
        
        

    def gini_impurity(self, y: np.array):
        # Compute the Gini impurity for a list of labels y
        uniq, counts = np.unique(y, return_counts=True)
        p = counts / y.size
        gini = 1 - np.sum(p**2)
        return gini


    def information_gain(self, criterion):
        # Compute the information gain from splitting y into y_left and y_right
        targets = self.split_node()['targets']
        if criterion == 'gini':
            impurity = self.gini_impurity(self.target)
            impurity_left = self.gini_impurity(targets[0])
            impurity_right = self.gini_impurity(targets[1])
        elif criterion == 'entropy':
            pass # Will implement later 
        elif criterion == 'error':
            pass # Will implement later 
        else:
            raise ValueError("Invalid criterion. Supported criteria are 'gini', 'entropy', and 'error'.")
        weight_left = len(targets[0]) / self.target.size
        weight_right = len(targets[1]) / self.target.size
        return impurity - (weight_left*impurity_left + weight_right*impurity_right)
    
    def is_pure(self, criterion):
        if self.info_gain <= 0.0:
            return True
        if np.unique(self.target).size == 1:
            return True
        else :
            return False




def best_split(node: Node, criterion):
    # Find the best feature and threshold to split the data
    rand_split_indx = list(range(node.features[0].size))
    features = [row for row in node.features]
    split_indx = -1
    score = 0.0
    threshold = None

    for i  in rand_split_indx:
        for row in features:
            node.split_index = i
            node.threshold = row[i]
            iG = node.information_gain(criterion=criterion)
            if iG > score:
                score = iG
                split_indx = i
                threshold = row[i]
    # Out of the loop, split_indx and thereshold correspond to the best feature indx abd threshold to split the node 

    return (split_indx, threshold, score)
    features = node.split_node()['features']
    targets = node.split_node()['targets']
    node.left = Node(features=features[0], target=targets[0], threshold=threshold, split_index=split_indx)
    node.right = Node(features= features[1], target= targets[1], threshold=threshold, split_index=split_indx)




def build_tree(node: Node, depth=0, max_depth=None, min_samples_split=2, criterion='gini'):
    # Recursively build the tree
    split_indx, threshold, score =  best_split(node, criterion)
    if split_indx != -1 and threshold is not None:
        node.split_index = split_indx
        node.threshold = threshold
        node.info_gain = score
    if stopping_criteria(node, depth, max_depth, node.target.size, min_samples_split, criterion):
        return node
    features = node.split_node()['features']
    targets = node.split_node()['targets']
    node.left = Node(features=features[0], target=targets[0])
    node.right = Node(features= features[1], target= targets[1])

    node.left = build_tree(node.left, depth+1, max_depth, min_samples_split, criterion)
    node.right = build_tree(node.right, depth+1, max_depth, min_samples_split, criterion)
    return node

    '''if not stopping_criteria(node.left, depth+1, max_depth, node.left.target.size, min_samples_split, criterion):
        build_tree(node.left, depth+1, max_depth, min_samples_split, criterion)
    if not stopping_criteria(node.right, depth+1, max_depth, node.right.target.size, min_samples_split):
        build_tree(node.right, depth+1, max_depth, min_samples_split, criterion)
    return'''



def stopping_criteria(node : Node, depth, max_depth, n_samples, min_samples_split, criterion = 'gini'):
    # Determine if we shuld stop splitting
    if node.is_pure(criterion) or depth >= max_depth or n_samples <= min_samples_split:
        return True
    else: return False

def predict_tree(node: Node, X):
    # Traverse the tree and make predictions for X
    if node.is_pure():
        uniq, counts = np.unique(node.target, return_counts=True)
        return uniq[counts.argmax()]
    if X[node.split_index] <= node.threshold:
        return predict_tree(node.left, X)
    else:
        return predict_tree(node.right, X)


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        node = Node(X, y)
        self.root = build_tree(node, 0, self.max_depth, self.min_samples_split, self.criterion)

    def predict(self, X):
        return [predict_tree(self.root, x) for x in X]


tree_model = DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train, y_train)


def print_tree(node: Node, d):
    print('Depth: ', d)
    print('\n')
    print([t for t in node.target])
    if node.left is not None:
        print_tree(node.left, d+1)
    if node.right is not None:
        print_tree(node.right, d+1)
    return

print_tree(tree_model.root, 0)