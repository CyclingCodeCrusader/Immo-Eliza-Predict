import pandas as pd

from functions.cleaner_functions import cosmetics_duplicates, subtype_consolidation, missing_values, unnecessary_columns
from functions.cleaner_functions import adjust_datatypes, mapping_subtypes, subtype_filter, plot_cleanup
from utils.utils import open_csv_as_dataframe, create_pkl_from_df, save_csv_pickle


df1 = open_csv_as_dataframe(default_path = r'data\after_scraping.csv')
print(df1.info())
exit
# Perform cosmetic cleaning and remove duplicates
df2 = cosmetics_duplicates(df1) # Remove stuff such as blank spaces, and remove duplicates

df3, steps = subtype_consolidation(df2)

# Dealing with missing values: assumptions and operations on the dataset
df4 = missing_values(df3)

df5 = unnecessary_columns(df4, columns_to_drop = ['room_count', 'sale_annuity', 'starting_price', 'transaction_type', 'has_garden'])

df6 = adjust_datatypes(df5)

df7, steps = mapping_subtypes(df6, steps)

df8, steps = subtype_filter(df7, steps)

plot_cleanup(steps, grp="subtype")

create_pkl_from_df(df8, file_path = r'data\after_datacleaning.pkl') # Give dataframe to save, and path to file

print("Summary of dataframe shapes:")
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)
print(df5.shape)
print(df6.shape)
print(df7.shape)
print(df8.shape)
