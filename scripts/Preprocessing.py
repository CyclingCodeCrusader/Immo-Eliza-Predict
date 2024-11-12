import pandas as pd
from tkinter import filedialog
from functions.utils import open_csv_as_dataframe, save_dataframe_to_pkl, open_pkl_as_dataframe, barplot
from functions.preprocessing_functions import map_province, assign_city_based_on_proximity_multiple_radii, outlier_handling_numerical, outlier_handling_categorical

df8 = open_pkl_as_dataframe(default_path = r'data\after_datacleaning.pkl')

df9 = map_province(df8)
barplot(df9,feature='province')

cities_data = {
    'city': ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liège', 'Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas'],
    'locality_latitude': [50.8503, 51.2211, 51.0543, 51.2093, 50.6050, 50.4674, 50.8798, 50.4542, 50.9403, 51.1449],
    'locality_longitude': [4.3517, 4.4120, 3.7174, 3.2240, 5.5797, 4.8712, 4.7033, 3.9514, 4.0364, 4.1525]
}
radius_list = [5, 10, 15]

df10 = assign_city_based_on_proximity_multiple_radii(df9, cities_data, radius_list)

# Part on outlier handling of categorical columns, via IQR or Z method and removing of outlier samples
df11 = outlier_handling_numerical(df10, num_data_cols=['price','net_habitable_surface','bedroom_count','facade_count']) #something wrong with 'land_surface', it gives a 0 when calcuating skew

# Part on outlier handling of categorical columns, via assigning rare values to other categories
df12 = outlier_handling_categorical(df11, cat_data_col='kitchen_type', category_map = {'Usa hyper equipped': 'Hyper equipped', 'Usa semi equipped': 'Semi equipped', 'Usa uninstalled':'Not installed', 'Usa installed':'Installed'})
df13 = outlier_handling_categorical(df12, cat_data_col='building_condition', category_map = {'To restore': 'To renovate'})
df14 = outlier_handling_categorical(df13, cat_data_col='epc', category_map = {'A+': 'A', 'A++': 'A', 'G':'F'})

save_dataframe_to_pkl(df14, file_path = r'data\after_analysis.pkl') # Give dataframe to save, and path to file

print(df9.shape)
print(df10.shape)
print(df11.shape)
print(df12.shape)
print(df13.shape)
print(df14.shape)

