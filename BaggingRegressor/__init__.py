import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_split = None
        best_mse = float('inf')
        
        # Find best split
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            
            for threshold in unique_values:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Skip if split is trivial
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                # Calculate weighted MSE
                left_mse = np.mean((y[left_mask] - np.mean(y[left_mask]))**2)
                right_mse = np.mean((y[right_mask] - np.mean(y[right_mask]))**2)
                weighted_mse = (len(y[left_mask]) * left_mse + len(y[right_mask]) * right_mse) / n_samples
                
                # Update best split
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_split = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        # If no good split found, return mean
        if best_split is None:
            return np.mean(y)
        
        # Recursive splitting
        left_subtree = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_subtree = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)
        
        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree
        }

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        
        feature = tree['feature']
        if x[feature] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

class BaggingRegressor:
    def __init__(self, base_model, n_estimators=10, max_depth=None):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        # Reset models before training
        self.models = []
        np.random.RandomState(seed=12)
        np.random.seed(12)
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Create and train model
            model = self.base_model(max_depth=self.max_depth)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        # Aggregate predictions from all models
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)

