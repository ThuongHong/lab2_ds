import numpy as np


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

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.n_features = X.shape[1]
        total_samples = len(y)

        # Calculate class priors
        for cls, count in zip(self.classes, class_counts):
            self.class_priors[cls] = count / total_samples

        # Store all unique values for each feature (for smoothing)
        for feature_idx in range(self.n_features):
            self.feature_categories[feature_idx] = np.unique(X[:, feature_idx])

        # Calculate feature likelihoods with Laplace smoothing
        for cls in self.classes:
            X_cls = X[y == cls]
            self.feature_likelihoods[cls] = {}

            for feature_idx in range(self.n_features):
                feature_values, value_counts = np.unique(
                    X_cls[:, feature_idx], return_counts=True
                )

                # Number of unique categories for this feature
                n_categories = len(self.feature_categories[feature_idx])

                # Total count for denominator with Laplace smoothing
                total_count = len(X_cls) + self.alpha * n_categories

                likelihoods = {}

                # Calculate likelihood for observed values
                for value, count in zip(feature_values, value_counts):
                    likelihoods[value] = (count + self.alpha) / total_count

                # Add smoothed probability for unseen values
                for value in self.feature_categories[feature_idx]:
                    if value not in likelihoods:
                        likelihoods[value] = self.alpha / total_count

                self.feature_likelihoods[cls][feature_idx] = likelihoods

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for cls in self.classes:
                class_prob = np.log(self.class_priors[cls])
                for feature_idx in range(self.n_features):
                    feature_value = x[feature_idx]
                    likelihoods = self.feature_likelihoods[cls].get(feature_idx, {})

                    # Get likelihood with Laplace smoothing for unseen values
                    if feature_value in likelihoods:
                        likelihood = likelihoods[feature_value]
                    else:
                        # For completely unseen values, use smoothing
                        n_categories = len(self.feature_categories.get(feature_idx, []))
                        total_count = sum(
                            [
                                self.class_priors[c] * len(X)
                                for c in self.classes
                                if c == cls
                            ]
                        )
                        likelihood = self.alpha / (
                            total_count + self.alpha * n_categories
                        )

                    class_prob += np.log(likelihood)
                class_probs[cls] = class_prob
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return np.array(predictions)
