def calculate_sensitivity_specificity(confusion_matrix):
    true_positives = confusion_matrix[1, 1]
    false_positives = confusion_matrix[0, 1]
    true_negatives = confusion_matrix[0, 0]
    false_negatives = confusion_matrix[1, 0]

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity