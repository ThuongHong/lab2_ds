import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter # Hữu ích để đếm tần suất

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
        plt.title(f'Target Distribution ({target_col_name})')
        plt.xlabel(target_col_name)
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_numerical_vs_target(num_data, target_data, num_col_name, target_col_name="Target", plot_type="box"):
        """
        Plots a side-by-side Box Plot for a numerical feature vs the target.

        Parameters:
        - num_data: array-like of the numerical feature values
        - target_data: array-like of the target values
        """
        plt.figure(figsize=(10, 6))
        
        if plot_type == "violin":
            sns.violinplot(x=target_data, y=num_data, palette="Pastel1", hue=target_data)
        else:
            sns.boxplot(x=target_data, y=num_data, palette="Pastel1", hue=target_data)
        
        plt.title(f'{plot_type.capitalize()} Plot: {num_col_name} by {target_col_name}')
        plt.xlabel(f'{target_col_name} (Target Class)')
        plt.ylabel(num_col_name)
        plt.show()

    @staticmethod
    def plot_categorical_vs_target(cat_data, target_data, cat_col_name, target_col_name="Target", percentage=False):
        """
        Plots a stacked bar chart for a categorical feature vs the target (showing proportions).
        
        Parameters:
        - cat_data: array-like of the categorical feature values
        - target_data: array-like of the target values
        """
        combined = list(zip(cat_data, target_data))
        cat_target_counts = Counter(combined)
        cat_counts = Counter(cat_data)
        
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
        plt.figure(figsize=(12, 7))
        for cls in target_classes:
            plt.bar(categories, proportions[cls], bottom=bottom, label=f'Target={cls}')
            bottom += np.array(proportions[cls])
        plt.title(f'Categorical Feature: {cat_col_name} vs {target_col_name}')
        plt.xlabel(cat_col_name)
        plt.ylabel('Proportion' if percentage else 'Count')
        plt.xticks(rotation=45)
        plt.legend(title=target_col_name)
        plt.show()