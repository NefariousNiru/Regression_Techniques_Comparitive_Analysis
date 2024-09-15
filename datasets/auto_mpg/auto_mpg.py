import os.path

import pandas as pd

def get_dataset():
    # Path to the file you uploaded
    file_path = os.path.join(os.path.dirname(__file__), 'auto-mpg.data')

    # Load the file using pandas (assuming it's whitespace-delimited as per earlier inspection)
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin',
                    'car_name']
    data = pd.read_csv(file_path, sep=r'\s+', names=column_names, na_values='?')
    return data


