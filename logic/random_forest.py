import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from view.plots import plot_confusion_matrix, plot_predictive_features_rf, print_evaluation_results
from logic.functions import evaluate_model


def random_forest(X, y, X_train, X_test, y_train, y_test):
    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf_classifier.predict(X_test)

    # model evaluation
    results = evaluate_model(y_test, y_pred)

    # get feature importances
    feature_names = X.columns.tolist()
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': rf_classifier.feature_importances_})

    return results, feature_importances

