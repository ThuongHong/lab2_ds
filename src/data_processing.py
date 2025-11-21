from collections import defaultdict, Counter
import numpy as np


class DataProcessor:
    @staticmethod
    def impute_by_mode(data, columns_to_impute=None):
        """
        Impute missing values by replacing them with the mode (most frequent value).

        Parameters:
        - data: 2D numpy array of the data
        - columns_to_impute: List of column indices to impute. If None, impute all columns with missing values

        Returns:
        - imputed_data: 2D numpy array with missing values replaced by mode
        - mode_values: Dictionary mapping column index to mode value used for imputation
        """
        imputed_data = data.copy()
        mode_values = {}

        if columns_to_impute is None:
            has_missing = np.any(data == "", axis=0)
            columns_to_impute = np.where(has_missing)[0].tolist()

        for idx in columns_to_impute:
            col_data = data[:, idx]
            non_missing = col_data[col_data != ""]

            if len(non_missing) == 0:
                continue

            counts = Counter(non_missing)
            mode = counts.most_common(1)[0][0]
            mode_values[idx] = mode

            imputed_data[:, idx] = np.where(col_data == "", mode, col_data)

        return imputed_data, mode_values

    @staticmethod
    def impute_as_category(data, columns_to_impute=None, missing_category="Missing"):
        """
        Treat missing values as a new category by replacing them with a specific value.

        Parameters:
        - data: 2D numpy array of the data
        - columns_to_impute: List of column indices to impute. If None, impute all columns with missing values
        - missing_category: String to use for the missing category (default: "Missing")

        Returns:
        - imputed_data: 2D numpy array with missing values replaced by missing_category
        - imputed_columns: List of column indices that were imputed
        """
        imputed_data = data.copy()
        imputed_columns = []

        if columns_to_impute is None:
            has_missing = np.any(data == "", axis=0)
            columns_to_impute = np.where(has_missing)[0].tolist()

        for idx in columns_to_impute:
            col_data = data[:, idx]

            if np.any(col_data == ""):
                imputed_data[:, idx] = np.where(
                    col_data == "", missing_category, col_data
                )
                imputed_columns.append(idx)

        return imputed_data, imputed_columns


class SMOTEN:
    """
    Generates synthetic samples for minority class to handle imbalanced datasets.
    """

    def __init__(self, k_neighbors=5, random_state=42):
        """
        Initialize SMOTE-N.

        Parameters:
        - k_neighbors: Number of nearest neighbors to use for generating synthetic samples
        - random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def _find_k_neighbors(self, X, sample_idx):
        """
        Find k nearest neighbors using vectorized Hamming distance.

        Parameters:
        - X: 2D array of samples
        - sample_idx: Index of the sample to find neighbors for

        Returns:
        - indices: Array of k nearest neighbor indices
        """
        sample = X[sample_idx]
        
        distances = np.sum(X != sample, axis=1).astype(float)
        distances[sample_idx] = np.inf
        
        if len(distances) > self.k_neighbors:
            neighbor_indices = np.argpartition(distances, self.k_neighbors)[:self.k_neighbors]
        else:
            neighbor_indices = np.argsort(distances)[:self.k_neighbors]
        
        return neighbor_indices

    def _generate_synthetic_sample(self, sample, neighbor, categorical_indices):
        """
        Generate a synthetic sample by randomly selecting features from sample and neighbor.

        Parameters:
        - sample: Original sample
        - neighbor: Neighbor sample
        - categorical_indices: Indices of categorical features

        Returns:
        - synthetic: New synthetic sample
        """
        synthetic = sample.copy()

        random_mask = np.random.random(len(categorical_indices)) < 0.5
        categorical_indices_arr = np.array(categorical_indices)
        synthetic[categorical_indices_arr[random_mask]] = neighbor[categorical_indices_arr[random_mask]]

        return synthetic

    def fit_resample(self, X, y, categorical_indices=None):
        """
        Apply SMOTE-N to balance the dataset.

        Parameters:
        - X: 2D numpy array of features (all categorical)
        - y: 1D numpy array of target labels
        - categorical_indices: List of indices of categorical features (if None, assumes all features are categorical)

        Returns:
        - X_resampled: Balanced feature array
        - y_resampled: Balanced target array
        """
        if categorical_indices is None:
            categorical_indices = list(range(X.shape[1]))

        unique_classes, class_counts = np.unique(y, return_counts=True)

        if len(unique_classes) < 2:
            print("Warning: Only one class found. No resampling needed.")
            return X, y

        max_count = np.max(class_counts)

        print(f"Original class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} samples")

        X_resampled = X.copy()
        y_resampled = y.copy()

        for cls, count in zip(unique_classes, class_counts):
            if count < max_count:
                class_indices = np.where(y == cls)[0]
                X_class = X[class_indices]
                n_synthetic = max_count - count

                print(f"Generating {n_synthetic} synthetic samples for class {cls}...")

                synthetic_samples = []

                for _ in range(n_synthetic):
                    sample_idx = np.random.randint(0, len(X_class))
                    neighbor_indices = self._find_k_neighbors(X_class, sample_idx)
                    neighbor_idx = np.random.choice(neighbor_indices)
                    synthetic = self._generate_synthetic_sample(
                        X_class[sample_idx], X_class[neighbor_idx], categorical_indices
                    )
                    synthetic_samples.append(synthetic)

                if synthetic_samples:
                    synthetic_samples = np.array(synthetic_samples)
                    X_resampled = np.vstack([X_resampled, synthetic_samples])
                    y_resampled = np.concatenate(
                        [y_resampled, np.full(len(synthetic_samples), cls)]
                    )

        print(f"\nResampled class distribution:")
        unique_classes_new, class_counts_new = np.unique(
            y_resampled, return_counts=True
        )
        for cls, count in zip(unique_classes_new, class_counts_new):
            print(f"  Class {cls}: {count} samples")

        return X_resampled, y_resampled
