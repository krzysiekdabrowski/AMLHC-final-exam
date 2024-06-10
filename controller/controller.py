from logic.data_processing import fetch_and_process_data
from logic.correlation import calculate_correlations
from view.plots import plot_correlations, print_correlations
from logic.regression import logistic_regression
from logic.random_forest import random_forest


def run():
    X, y = fetch_and_process_data()

    correlations = calculate_correlations(X, y)
    plot_correlations(correlations)

    # logistic_regression(X, y)
    random_forest(X, y)
    # DNN(X, y)

