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
            # Find all columns with missing values
            columns_to_impute = []
            for i in range(data.shape[1]):
                if np.any(data[:, i] == ""):
                    columns_to_impute.append(i)

        for idx in columns_to_impute:
            col_data = data[:, idx]

            # Get non-missing values
            non_missing = col_data[col_data != ""]

            if len(non_missing) == 0:
                # If all values are missing, skip this column
                continue

            # Calculate mode using Counter
            counts = Counter(non_missing)
            mode = counts.most_common(1)[0][0]
            mode_values[idx] = mode

            # Replace missing values with mode
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
            # Find all columns with missing values
            columns_to_impute = []
            for i in range(data.shape[1]):
                if np.any(data[:, i] == ""):
                    columns_to_impute.append(i)

        for idx in columns_to_impute:
            col_data = data[:, idx]

            # Replace missing values with the missing category
            if np.any(col_data == ""):
                imputed_data[:, idx] = np.where(
                    col_data == "", missing_category, col_data
                )
                imputed_columns.append(idx)

        return imputed_data, imputed_columns
