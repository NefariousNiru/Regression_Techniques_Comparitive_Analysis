import os
from util import pre_processing, load, eda

'''
Fetches the data
Performs Exploratory Analysis (Datatypes, Correlation Matrix, Missing Values, Group Analysis, Descriptive Statistics)
Replaces column names to more meaningful ones if needed
Plots outliers
Returns a dataframe with zero n/a values and all columns
'''
def get_dataset():
    columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    file_path = os.path.join(os.path.dirname(__file__), 'boston.csv')
    data = load.dataset(file_path, columns, None,1)

    print("____________________________")
    print("Exploratory Data Analysis for Boston Dataset")

    # Print data types
    pre_processing.print_dtypes("Boston Housing", data)

    # Visualize missing data
    eda.visualize_missing_values(data)

    # Drop n/a values
    data = pre_processing.clean_data(data)

    # Print correlation matrix
    eda.print_correlation_matrix(data)

    # Group analysis of data (you may want to specify which columns are categorical for group analysis)
    eda.group_analysis(data, ['CHAS', 'RAD'], 'MEDV')  # Example using 'CHAS' and 'RAD' as categorical columns

    # Describe the data
    eda.print_descriptive_statistics(data)

    # Print Outliers
    eda.plot_outliers(data)

    print("____________________________")

    return data
