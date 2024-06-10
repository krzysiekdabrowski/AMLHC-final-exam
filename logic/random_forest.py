import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo
from view.plots import plot_confusion_matrix, plot_predictive_features_rf
from logic.functions import calculate_sensitivity_specificity


def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)


    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Calculate sensitivity and specificity
    sensitivity, specificity = calculate_sensitivity_specificity(conf_matrix)

    # Classification report
    class_report = classification_report(y_test, y_pred)


    print("Accuracy:", np.round(accuracy, 2))
    print("Sensitivity:", np.round(sensitivity, 2))
    print("Specificity:", np.round(specificity, 2))
    print("Confusion Matrix:")
    print(conf_matrix)

    print("Classification Report:")
    print(class_report)

    # Get feature importances
    feature_names = X.columns.tolist()
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': rf_classifier.feature_importances_})

    # Plot predictive features
    plot_predictive_features_rf(feature_importances)
