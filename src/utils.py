import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred, positive_class):
    true_positives = np.sum((y_pred == positive_class) & (y_true == positive_class))
    predicted_positives = np.sum(y_pred == positive_class)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall(y_true, y_pred, positive_class):
    true_positives = np.sum((y_pred == positive_class) & (y_true == positive_class))
    actual_positives = np.sum(y_true == positive_class)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def f1_score(y_true, y_pred, positive_class):
    prec = precision(y_true, y_pred, positive_class)
    rec = recall(y_true, y_pred, positive_class)
    if (prec + rec) == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def classification_report(y_true, y_pred):
    classes = np.unique(y_true)
    report = {}
    for cls in classes:
        report[cls] = {
            "precision": precision(y_true, y_pred, cls),
            "recall": recall(y_true, y_pred, cls),
            "f1_score": f1_score(y_true, y_pred, cls),
        }
    report["accuracy"] = accuracy(y_true, y_pred)
    
    # print report
    for cls in classes:
        print(
            f"Class {cls}: Precision: {report[cls]['precision']:.4f}, "
            f"Recall: {report[cls]['recall']:.4f}, F1-Score: {report[cls]['f1_score']:.4f}"
        )
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    
    return report
