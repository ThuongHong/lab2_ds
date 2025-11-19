import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter


class EDAVisualizer:
    @staticmethod
    def plot_target_distribution(target_data, target_col_name="Target"):
        """
        Plots the distribution of the target variable (Count Plot).

        Parameters:
        - target_data: array-like (e.g., list or numpy array) of the target values
        """
        plt.figure(figsize=(8, 5))

        counts = Counter(target_data)
        labels = list(counts.keys())
        values = list(counts.values())

        # Truncate labels
        truncated_labels = [
            str(label)[:8] + ".." if len(str(label)) > 8 else str(label)
            for label in labels
        ]

        sns.barplot(x=truncated_labels, y=values, palette="viridis", hue=values)
        plt.title(f"Target Distribution ({target_col_name})")
        plt.xlabel(target_col_name)
        plt.ylabel("Count")
        plt.show()

    @staticmethod
    def plot_numerical_vs_target(
        num_data, target_data, num_col_name, target_col_name="Target", plot_type="box"
    ):
        """
        Plots a side-by-side Box Plot for a numerical feature vs the target.

        Parameters:
        - num_data: array-like of the numerical feature values
        - target_data: array-like of the target values
        """
        plt.figure(figsize=(10, 6))

        # Truncate target labels
        unique_targets = sorted(list(set(target_data)))
        truncated_targets = [
            str(t)[:8] + ".." if len(str(t)) > 8 else str(t) for t in unique_targets
        ]
        target_mapping = {
            old: new for old, new in zip(unique_targets, truncated_targets)
        }
        truncated_target_data = [target_mapping[t] for t in target_data]

        # Convert numerical data to float and truncate y-axis labels
        num_data_float = num_data.astype(float)

        if plot_type == "violin":
            sns.violinplot(
                x=truncated_target_data,
                y=num_data_float,
                palette="Pastel1",
                hue=truncated_target_data,
            )
        else:
            sns.boxplot(
                x=truncated_target_data,
                y=num_data_float,
                palette="Pastel1",
                hue=truncated_target_data,
            )

        plt.title(f"{plot_type.capitalize()} Plot: {num_col_name} by {target_col_name}")
        plt.xlabel(f"{target_col_name} (Target Class)")
        plt.ylabel(num_col_name)

        # Format y-axis labels to show fewer decimal places
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3f}"))

        plt.show()

    @staticmethod
    def plot_categorical_vs_target(
        cat_data, target_data, cat_col_names, target_col_name="Target", percentage=False
    ):
        """
        Plots multiple stacked bar charts for categorical features vs the target using subplots.
        Each subplot corresponds to a unique categorical feature.

        Parameters:
        - cat_data: 2D array-like of the categorical feature values (each column is a feature)
        - target_data: array-like of the target values
        - cat_col_names: List of categorical feature names
        - target_col_name: Name of the target variable
        - percentage: Whether to show proportions instead of counts
        """
        num_features = len(cat_col_names)
        num_rows = (num_features + 2) // 3  # 3 plots per row

        fig, axes = plt.subplots(
            num_rows, 3, figsize=(18, 5 * num_rows), constrained_layout=True
        )
        axes = axes.flatten()  # Flatten to easily iterate over

        for i, (cat_col_name, cat_col_data) in enumerate(
            zip(cat_col_names, cat_data.T)
        ):
            combined = list(zip(cat_col_data, target_data))
            cat_target_counts = Counter(combined)
            cat_counts = Counter(cat_col_data)

            categories = sorted(cat_counts.keys())
            target_classes = sorted(list(set(target_data)))

            proportions = {cls: [] for cls in target_classes}

            if percentage:
                for cat in categories:
                    total = cat_counts[cat]
                    for cls in target_classes:
                        count = cat_target_counts.get((cat, cls), 0)
                        proportions[cls].append(count / total if total > 0 else 0)
            else:
                for cat in categories:
                    for cls in target_classes:
                        count = cat_target_counts.get((cat, cls), 0)
                        proportions[cls].append(count)

            # Truncate long labels
            truncated_labels = [
                str(cat)[:6] + ".." if len(str(cat)) > 6 else str(cat)
                for cat in categories
            ]

            bottom = np.zeros(len(categories))
            for cls in target_classes:
                axes[i].bar(
                    range(len(categories)),
                    proportions[cls],
                    bottom=bottom,
                    label=f"Target={cls}",
                )
                bottom += np.array(proportions[cls])

            axes[i].set_title(f"{cat_col_name} vs {target_col_name}", fontsize=10)
            axes[i].set_xlabel(cat_col_name, fontsize=9)
            axes[i].set_ylabel("Proportion" if percentage else "Count", fontsize=9)

            # Adjust tick frequency and size based on number of categories
            if len(categories) > 50:
                # Show every 5th label for very large categories
                tick_indices = list(range(0, len(categories), 5))
                axes[i].set_xticks(tick_indices)
                axes[i].set_xticklabels(
                    [truncated_labels[idx] for idx in tick_indices],
                    rotation=90,
                    ha="right",
                    fontsize=5,
                )
            elif len(categories) > 20:
                # Show every 2nd label for medium categories
                tick_indices = list(range(0, len(categories), 2))
                axes[i].set_xticks(tick_indices)
                axes[i].set_xticklabels(
                    [truncated_labels[idx] for idx in tick_indices],
                    rotation=90,
                    ha="right",
                    fontsize=6,
                )
            else:
                axes[i].set_xticks(range(len(categories)))
                axes[i].set_xticklabels(
                    truncated_labels, rotation=45, ha="right", fontsize=8
                )

            axes[i].legend(title=target_col_name, fontsize=8)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.show()

    @staticmethod
    def plot_categorical_correlation(
        cat_col1_data,
        cat_col2_data,
        cat_col1_name="Category 1",
        cat_col2_name="Category 2",
    ):
        """
        Plots a heatmap showing the correlation between two categorical features.

        Parameters:
        - cat_col1_data: array-like of the first categorical feature values
        - cat_col2_data: array-like of the second categorical feature values
        - cat_col1_name: Name of the first categorical feature
        - cat_col2_name: Name of the second categorical feature
        """
        combined = list(zip(cat_col1_data, cat_col2_data))
        pair_counts = Counter(combined)

        categories1 = sorted(list(set(cat_col1_data)))
        categories2 = sorted(list(set(cat_col2_data)))

        correlation_matrix = np.zeros((len(categories1), len(categories2)))

        for i, cat1 in enumerate(categories1):
            for j, cat2 in enumerate(categories2):
                correlation_matrix[i, j] = pair_counts.get((cat1, cat2), 0)

        # Truncate labels for better readability
        truncated_cat1 = [
            str(cat)[:6] + ".." if len(str(cat)) > 6 else str(cat)
            for cat in categories1
        ]
        truncated_cat2 = [
            str(cat)[:6] + ".." if len(str(cat)) > 6 else str(cat)
            for cat in categories2
        ]

        # Adjust figure size based on number of categories
        fig_width = max(12, len(categories2) * 0.3)
        fig_height = max(10, len(categories1) * 0.3)

        plt.figure(figsize=(fig_width, fig_height))

        # Only show annotations if not too many categories
        show_annot = len(categories1) * len(categories2) < 500

        # Use log scale for better visualization of sparse data
        # Add 1 to avoid log(0) and apply log transformation
        correlation_matrix_log = np.log1p(correlation_matrix)

        sns.heatmap(
            correlation_matrix_log,
            annot=show_annot,
            fmt=".1f",
            cmap="YlGnBu",
            xticklabels=truncated_cat2,
            yticklabels=truncated_cat1,
            cbar_kws={"label": "Count (log scale)"},
            vmin=0,
            vmax=np.percentile(correlation_matrix_log[correlation_matrix_log > 0], 95),
            linewidths=0.5,
            linecolor="lightgray",
        )

        plt.title(
            f"Correlation between {cat_col1_name} and {cat_col2_name}", fontsize=12
        )
        plt.xlabel(cat_col2_name, fontsize=10)
        plt.ylabel(cat_col1_name, fontsize=10)
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_missing_values_heatmap(data, headers):
        """
        Plots a heatmap showing the nullity correlation between features.

        Nullity correlation ranges from -1 to 1:
        -1: Exact negative correlation (if one is present, other is definitely absent)
        0: No correlation (values present/absent have no effect on one another)
        1: Exact positive correlation (if one is present, other is definitely present)

        Parameters:
        - data: 2D numpy array of the data
        - headers: List of column names
        """
        n_features = len(headers)
        n_samples = len(data)

        # Create binary matrix: 1 if present, 0 if missing
        present_matrix = (data != "").astype(int)

        # Calculate correlation matrix for nullity
        correlation_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                # Get presence indicators for both features
                col_i = present_matrix[:, i]
                col_j = present_matrix[:, j]

                # Calculate Pearson correlation coefficient
                mean_i = np.mean(col_i)
                mean_j = np.mean(col_j)

                if np.std(col_i) == 0 or np.std(col_j) == 0:
                    correlation_matrix[i, j] = 0
                else:
                    cov = np.mean((col_i - mean_i) * (col_j - mean_j))
                    correlation_matrix[i, j] = cov / (np.std(col_i) * np.std(col_j))

        # Truncate column names
        truncated_cols = [
            str(col)[:15] + ".." if len(str(col)) > 15 else str(col) for col in headers
        ]

        # Create mask for upper triangle and diagonal
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=0)

        plt.figure(figsize=(10, 5))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap="RdYlGn",
            xticklabels=truncated_cols,
            yticklabels=truncated_cols,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Nullity Correlation"},
            vmin=-1,
            vmax=1,
            center=0,
            # linewidths=0.5,
            # linecolor='lightgray',
        )
        plt.title("Heatmap of Nullity Correlation", fontsize=12)
        plt.xlabel("Features", fontsize=10)
        plt.ylabel("Features", fontsize=10)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cramers_v_heatmap(data, headers):
        """
        Plots a heatmap showing Cramér's V correlation between categorical features.

        Cramér's V measures association between categorical variables, ranging from 0 to 1:
        0: No association between variables
        1: Perfect association between variables

        Parameters:
        - data: 2D numpy array of the data
        - headers: List of column names
        """

        def cramers_v(x, y):
            """Calculate Cramér's V statistic for categorical-categorical association."""
            # Remove missing values
            mask = (x != "") & (y != "")
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) == 0:
                return 0

            # Get unique categories
            categories_x, x_indices = np.unique(x_clean, return_inverse=True)
            categories_y, y_indices = np.unique(y_clean, return_inverse=True)

            # Create contingency table using bincount
            contingency = np.bincount(
                x_indices * len(categories_y) + y_indices,
                minlength=len(categories_x) * len(categories_y),
            ).reshape(len(categories_x), len(categories_y))

            # Calculate chi-square statistic
            row_sums = contingency.sum(axis=1, keepdims=True)
            col_sums = contingency.sum(axis=0, keepdims=True)
            total = contingency.sum()

            if total == 0:
                return 0

            expected = (row_sums @ col_sums) / total
            expected = np.where(expected == 0, 1e-10, expected)

            chi2 = np.sum((contingency - expected) ** 2 / expected)

            # Calculate Cramér's V
            min_dim = min(len(categories_x) - 1, len(categories_y) - 1)

            if min_dim == 0:
                return 0

            cramers = np.sqrt(chi2 / (total * min_dim))
            return min(cramers, 1.0)

        n_features = len(headers)
        correlation_matrix = np.zeros((n_features, n_features))

        print("Calculating Cramér's V correlation matrix...")
        # Calculate Cramér's V for all pairs
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    v = cramers_v(data[:, i], data[:, j])
                    correlation_matrix[i, j] = v
                    correlation_matrix[j, i] = v

        # Truncate column names
        truncated_cols = [
            str(col)[:12] + ".." if len(str(col)) > 12 else str(col) for col in headers
        ]

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

        plt.figure(figsize=(10, 5))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap="YlOrRd",
            xticklabels=truncated_cols,
            yticklabels=truncated_cols,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Cramér's V"},
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
        )
        plt.title(
            "Heatmap of Cramér's V Correlation (Categorical Features)", fontsize=14
        )
        plt.xlabel("Features", fontsize=11)
        plt.ylabel("Features", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.show()


class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(confusion_matrix, class_names):
        """
        Plots a confusion matrix heatmap.

        Parameters:
        - confusion_matrix: 2D numpy array representing the confusion matrix
        - class_names: List of class names corresponding to the matrix indices
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
            linewidths=0.5,
            linecolor="lightgray",
        )
        plt.title("Confusion Matrix", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.show()
