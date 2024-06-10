import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # standardize the features (not necessary for random forest, but done anyway for all models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test