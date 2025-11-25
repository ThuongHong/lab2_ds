import numpy as np
from collections import defaultdict


class CategoricalNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize Categorical Naive Bayes with Laplace smoothing.

        Parameters:
        - alpha: Smoothing parameter (default=1.0 for Laplace smoothing)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.classes = None
        self.n_features = 0
        self.feature_categories = {}
        self.class_counts = {}

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.n_features = X.shape[1]
        total_samples = len(y)

        for cls, count in zip(self.classes, class_counts):
            self.class_priors[cls] = count / total_samples
            self.class_counts[cls] = count

        for feature_idx in range(self.n_features):
            self.feature_categories[feature_idx] = np.unique(X[:, feature_idx])

        for cls in self.classes:
            X_cls = X[y == cls]
            self.feature_likelihoods[cls] = {}

            for feature_idx in range(self.n_features):
                feature_values, value_counts = np.unique(
                    X_cls[:, feature_idx], return_counts=True
                )

                n_categories = len(self.feature_categories[feature_idx])
                total_count = len(X_cls) + self.alpha * n_categories

                likelihoods = {}

                for value, count in zip(feature_values, value_counts):
                    likelihoods[value] = (count + self.alpha) / total_count

                for value in self.feature_categories[feature_idx]:
                    if value not in likelihoods:
                        likelihoods[value] = self.alpha / total_count

                self.feature_likelihoods[cls][feature_idx] = likelihoods

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        log_probs = np.zeros((n_samples, n_classes))
        
        for idx, cls in enumerate(self.classes):
            prior = self.class_priors[cls]
            log_probs[:, idx] = np.log(np.clip(prior, 1e-10, None))
        
        for feature_idx in range(self.n_features):
            feature_column = X[:, feature_idx]
            
            for cls_idx, cls in enumerate(self.classes):
                likelihoods = self.feature_likelihoods[cls][feature_idx]
                n_categories = len(self.feature_categories[feature_idx])
                total_count = self.class_counts[cls] + self.alpha * n_categories
                default_likelihood = self.alpha / total_count
                
                likelihood_vec = np.array([
                    likelihoods.get(val, default_likelihood) 
                    for val in feature_column
                ])
                
                likelihood_vec = np.clip(likelihood_vec, 1e-10, None)
                log_likelihood_vec = np.log(likelihood_vec)
                
                log_probs[:, cls_idx] = np.einsum('i,i->i', 
                    np.ones(n_samples), 
                    log_probs[:, cls_idx]) + log_likelihood_vec
        
        predicted_indices = np.argmax(log_probs, axis=1)
        predictions = self.classes[predicted_indices]
        
        return predictions
