import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logic.data import split_dataset, fetch_and_process_data
from logic.deep_neural_network import deep_neural_network
from logic.functions import calculate_correlations
from view.plots import plot_correlations, print_evaluation_results, plot_confusion_matrix, plot_predictive_features, \
    plot_predictive_features_rf, plot_predictive_features_dnn
from logic.regression import logistic_regression
from logic.random_forest import random_forest


def run():
    X, y = fetch_and_process_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    #logistic_regression_controller(X, y, X_train, X_test, y_train, y_test)
    #random_forest_controller(X, y, X_train, X_test, y_train, y_test)
    DNN_controller(X, y, X_train, X_test, y_train, y_test)


def logistic_regression_controller(X, y, X_train, X_test, y_train, y_test):
    lr_results, sorted_coefficients = logistic_regression(X, y, X_train, X_test, y_train, y_test)
    print_evaluation_results(lr_results)
    plot_confusion_matrix(lr_results["matrix"])
    plot_predictive_features(sorted_coefficients)


def random_forest_controller(X, y, X_train, X_test, y_train, y_test):
    rf_results, importances = random_forest(X, y, X_train, X_test, y_train, y_test)
    print_evaluation_results(rf_results)
    plot_confusion_matrix(rf_results["matrix"])
    plot_predictive_features_rf(importances)


def DNN_controller(X, y, X_train, X_test, y_train, y_test):
    dnn_results = deep_neural_network(X, y, X_train, X_test, y_train, y_test)
    print_evaluation_results(dnn_results)
    plot_confusion_matrix(dnn_results["matrix"])
    #feature_names = X.columns.tolist()
    #plot_predictive_features_dnn(importances, feature_names)


def correlations(X, y):
    # find the strongest correlations between features and target variable
    correlations = calculate_correlations(X, y)
    plot_correlations(correlations)
