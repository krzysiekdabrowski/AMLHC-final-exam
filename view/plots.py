import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
