import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def print_correlation_matrix(data):
    correlation_matrix = data.corr()
    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Auto MPG Dataset")
    plt.show()


def visualize_missing_values(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

def print_descriptive_statistics(data):
    print()
    with pd.option_context('display.max_columns', None):
        print("Descriptive Statistics:")
        desc = data.describe().T
        skewness = data.skew().rename('skewness')
        kurtosis = data.kurt().rename('kurtosis')
        desc['skewness'] = skewness
        desc['kurtosis'] = kurtosis
        print(desc)


def group_analysis(data, x_columns, y):
    for column in x_columns:
        print(f"\nAverage {y} by {column}:")
        print(data.groupby(column)[y].mean())


def plot_outliers(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Create subplots: n rows based on number of numeric columns
    num_columns = len(numeric_columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(5 * num_columns, 6))

    # If there's only one numeric column, make axes a list to iterate over
    if num_columns == 1:
        axes = [axes]

    # Plot each numeric column as a boxplot in a subplot
    for i, column in enumerate(numeric_columns):
        sns.boxplot(data[column], ax=axes[i])
        axes[i].set_title(f'{column}')

    # Adjust layout
    plt.tight_layout()
    plt.show()

