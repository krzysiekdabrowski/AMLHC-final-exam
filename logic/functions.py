from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from scipy.stats import pointbiserialr


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    sensitivity, specificity = calculate_sensitivity_specificity(conf_matrix)
    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "matrix": conf_matrix,
        "report": class_report
    }
    return metrics

def calculate_sensitivity_specificity(confusion_matrix):
    true_positives = confusion_matrix[1, 1]
    false_positives = confusion_matrix[0, 1]
    true_negatives = confusion_matrix[0, 0]
    false_negatives = confusion_matrix[1, 0]

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity


def calculate_correlations(X, y):
    correlations = {}

    for column in X.columns:
        feature = X[column]  # Ensure feature is a Series
        pointbiserial_corr, _ = pointbiserialr(feature, y)
        correlations[column] = pointbiserial_corr

    # Convert to DataFrame for better readability
    pointbiserial_df = pd.DataFrame.from_dict(correlations, orient='index',
                                              columns=['Point-Biserial Correlation'])

    return pointbiserial_df