import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(confusion_matrix):
    labels = ['Negative', 'Positive']
    plt.figure(figsize=(8, 6))

    sns.set(font_scale=15 / 10)
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels, vmin=0,
                vmax=np.max(confusion_matrix) * 2)


    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_predictive_features(sorted_coefficients):
    # Visualizing the top predictive features
    plt.figure(figsize=(12, 10))
    sns.barplot(y='Coefficient', x='Feature', hue='Coefficient', data=sorted_coefficients, palette='viridis', legend=False)
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title('Top Predictive Features in Logistic Regression')
    plt.xticks(rotation=90)
    plt.show()


def plot_predictive_features_rf(feature_importances):
    # Sort feature importances
    sorted_features = feature_importances.sort_values(by='Importance', ascending=False)

    # Visualizing the top predictive features
    plt.figure(figsize=(12, 10))
    sns.barplot(y='Importance', x='Feature', hue='Importance', data=sorted_features, palette='viridis', legend=False)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Top Predictive Features in Random Forest')
    plt.xticks(rotation=90)
    plt.show()


def plot_predictive_features_dnn(importances, feature_names):
    sorted_importances = sorted(importances, reverse=True)
    sorted_feature_names = [feature_names[i] for i in np.argsort(importances)[::-1]]

    # Creating a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': sorted_feature_names,
        'Importance': sorted_importances
    })

    # Print the feature importances
    print(importance_df)

    # Plotting
    plt.figure(figsize=(12, 10))
    sns.barplot(y='Feature', x='Importance', hue='Feature', data=importance_df, palette='viridis', dodge=False)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top Predictive Features in DNN')
    plt.xticks(rotation=90)
    plt.legend([], [], frameon=False)  # Remove the legend
    plt.show()


def plot_correlations(correlation_df):
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlation_df.index, y=correlation_df.iloc[:, 0], data=correlation_df)
    plt.xticks(rotation=90)
    plt.title('Point-Biserial Correlations Between Features and Status Variable')
    plt.show()


def print_correlations(correlation_df):
    print("Point-Biserial Correlations:")
    print(correlation_df.sort_values(by='Point-Biserial Correlation', ascending=False))


def display_model_results(results):
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.2f}")


def print_evaluation_results(results):
    print("Accuracy:", np.round(results["accuracy"], 2))
    print("Sensitivity:", np.round(results["sensitivity"], 2))
    print("Specificity:", np.round(results["specificity"], 2))
    print("Confusion Matrix:\n", results["matrix"])
    print("Classification Report:\n", results["report"])
