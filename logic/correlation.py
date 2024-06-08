import pandas as pd
from scipy.stats import pointbiserialr

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

