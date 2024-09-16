import os.path
from util import pre_processing, load, eda


'''
Fetches the data
Performs Exploratory Analysis (Datatypes, Correlation Matrix, Missing Values, Group Analysis, Descriptive Statistics)
Replaces car names to car brands (more meaningful)
Plots outliers
Returns a dataframe with zero n/a values and all columns
'''
def get_dataset():
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    file_path = os.path.join(os.path.dirname(__file__), 'auto-mpg.data')
    data = load.dataset(file_path, columns, r'\s+', 0)

    print("____________________________")
    print("Exploratory Data Analysis for Auto MPG")

    # Print data types
    pre_processing.print_dtypes("Auto MPG", data)

    # Visualize missing data
    eda.visualize_missing_values(data)

    # Drop n/a values
    data = pre_processing.clean_data(data)

    # Replace car names to brand names (more meaningful_feature)
    data = replace_car_names_to_brand(data)

    # Print correlation matrix without brand column
    eda.print_correlation_matrix(data.drop(columns='brand'))

    # Group analysis of data
    eda.group_analysis(data, ['brand', 'cylinders', 'model_year', 'origin'], 'mpg')

    # Describe the data
    eda.print_descriptive_statistics(data.drop(columns=['brand']))

    # Print Outliers
    eda.plot_outliers(data)

    print("____________________________")

    return data


def replace_car_names_to_brand(data):
    data.loc[:,'brand'] = data.loc[:,'car_name'].transform(lambda x: x.split()[0])
    data = data.drop(columns=['car_name'])
    return data