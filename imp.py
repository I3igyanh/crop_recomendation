import pandas as pd

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Display the first few rows of the dataset
print(data.head())
print(data.info())
