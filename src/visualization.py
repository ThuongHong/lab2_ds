import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter


class EDAPlotter:
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

        sns.barplot(x=labels, y=values, palette="viridis", hue=values)
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

        if plot_type == "violin":
            sns.violinplot(
                x=target_data, y=num_data, palette="Pastel1", hue=target_data
            )
        else:
            sns.boxplot(x=target_data, y=num_data, palette="Pastel1", hue=target_data)

        plt.title(f"{plot_type.capitalize()} Plot: {num_col_name} by {target_col_name}")
        plt.xlabel(f"{target_col_name} (Target Class)")
        plt.ylabel(num_col_name)
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

        for i, (cat_col_name, cat_col_data) in enumerate(zip(cat_col_names, cat_data.T)):
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

            bottom = np.zeros(len(categories))
            for cls in target_classes:
                axes[i].bar(categories, proportions[cls], bottom=bottom, label=f"Target={cls}")
                bottom += np.array(proportions[cls])

            axes[i].set_title(f"{cat_col_name} vs {target_col_name}")
            axes[i].set_xlabel(cat_col_name)
            axes[i].set_ylabel("Proportion" if percentage else "Count")
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].legend(title=target_col_name)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.show()
