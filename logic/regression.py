import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo
from view.plots import plot_confusion_matrix, plot_predictive_features
from logic.functions import calculate_sensitivity_specificity


def logistic_regression(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the logistic regression model
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = log_reg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sensitivity, specificity = calculate_sensitivity_specificity(conf_matrix)

    class_report = classification_report(y_test, y_pred)

    # Print results
    print("Accuracy:", np.round(accuracy, 2))
    print("Sensitivity:", np.round(sensitivity, 2))
    print("Specificity:", np.round(specificity, 2))
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Plot results
    plot_confusion_matrix(conf_matrix)

    # Finding most predictive feature with coefficients
    feature_names = X.columns.tolist()
    coefficients_df = pd.DataFrame({'Coefficient': log_reg.coef_[0], 'Feature': feature_names})
    sorted_coefficients = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)
    print(sorted_coefficients)
    plot_predictive_features(sorted_coefficients)
