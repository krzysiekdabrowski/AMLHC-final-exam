import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_and_process_data():
    parkinsons = fetch_ucirepo(id=174)
    X = parkinsons.data.features
    y = parkinsons.data.targets['status']

    # Handle duplicate column names by renaming them
    X.columns = [f"{col}_2" if X.columns.duplicated()[i] else col for i, col in enumerate(X.columns)]

    # Verify that y is a pandas Series
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()

    return X, y
