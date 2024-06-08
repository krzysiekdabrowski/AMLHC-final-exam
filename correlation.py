import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy.stats import pointbiserialr

# Fetch the dataset
parkinsons = fetch_ucirepo(id=174)

# Extract features and targets
X = parkinsons.data.features
y = parkinsons.data.targets['status']  # Ensure 'status' is the target column name

# Verify that y is a pandas Series
if isinstance(y, pd.DataFrame):
    y = y.squeeze()

# Handle duplicate column names by renaming them
X.columns = [f"{col}_2" if X.columns.duplicated()[i] else col for i, col in enumerate(X.columns)]



# Initialize a dictionary to store point-biserial correlation results
pointbiserial_correlations = {}

# Calculate point-biserial correlations
for column in X.columns:
    feature = X[column]  # Ensure feature is a Series

    # Calculate point-biserial correlation
    pointbiserial_corr, _ = pointbiserialr(feature, y)
    pointbiserial_correlations[column] = pointbiserial_corr

# Convert to DataFrame for better readability
pointbiserial_df = pd.DataFrame.from_dict(pointbiserial_correlations, orient='index',
                                          columns=['Point-Biserial Correlation'])

# Print sorted results
print("Point-Biserial Correlations:")
print(pointbiserial_df.sort_values(by='Point-Biserial Correlation', ascending=False))

# Plotting the correlations
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlations(correlation_df, title):
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlation_df.index, y=correlation_df.iloc[:, 0], data=correlation_df)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()


plot_correlations(pointbiserial_df, 'Point-Biserial Correlation with Status')