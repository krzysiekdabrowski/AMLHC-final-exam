from logic.data_processing import fetch_and_process_data
from logic.correlation import calculate_correlations
from view.plots import plot_correlations, print_correlations


def run():
    X, y = fetch_and_process_data()

    correlations = calculate_correlations(X, y)
    plot_correlations(correlations)
    #print_correlations(correlations)