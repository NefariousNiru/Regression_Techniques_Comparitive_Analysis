import os
from util import pre_processing, load, eda

'''
Fetches the data
Performs Exploratory Analysis (Datatypes, Correlation Matrix, Missing Values, Group Analysis, Descriptive Statistics)
Replaces car names to car brands (more meaningful)
Plots outliers
Returns a dataframe with zero n/a values and all columns
'''
def get_dataset():
    columns = ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"]
    file_path = os.path.join(os.path.dirname(__file__), 'forestfires.csv')
    data = load.dataset(file_path, columns, None, 1)

    print("____________________________")
    print("Exploratory Data Analysis for Forest Fires")

    # Print data types
    pre_processing.print_dtypes("Forest Fires", data)

    # Visualize missing data
    eda.visualize_missing_values(data)

    # Drop n/a values
    data = pre_processing.clean_data(data)

    # Print correlation matrix without brand column
    eda.print_correlation_matrix(data.drop(columns=['month', 'day']), 'Forest Fires')

    # Group analysis of data
    eda.group_analysis(data, ['month', 'day', 'rain', 'X', 'Y'], 'area')

    # Describe the data
    eda.print_descriptive_statistics(data.drop(columns=['month', 'day']))

    # Print Outliers
    eda.plot_outliers(data)

    # One hot encoding
    data = pre_processing.encode_one_hot(['month', 'day'], data)

    print("____________________________")

    return data
