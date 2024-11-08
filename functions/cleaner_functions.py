
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


def cosmetics_duplicates(df):

    # Edit text in the columns
    text_edit_columns = ['subtype', 'transaction_type', 'kitchen_type', 'building_condition' ]

    for column in text_edit_columns:
        df[column] = df[column].astype(str)  # Ensure the column is treated as string
        df[column] = df[column].str.replace('_', ' ').str.capitalize() #Replace "_" to space, editing text (first capital letter, rest lower case)

    #Edit text of cities and street names
    names_edit_columns = ['locality_name'] #, 'street' street not scraped

    for column in names_edit_columns:
        df[column] = df[column].astype(str)  # Ensure the column is treated as string
        df[column] = df[column].str.title() # Editing text (first capital letters, rest lower case)

    # Remove zip code from brackets, e.g. "Tielt (8700)" -> "Tielt"
    df['locality_name'] = df['locality_name'].str.replace(r"\s*\(\d+\)", "", regex=True)

    # Edit the numbers in columns
    # Selecting the columns to change
    number_edit_columns = ['bedroom_count', 'net_habitable_surface', 'land_surface', 'facade_count', 'price', 'terrace_area', 'garden_area']

    # Converting the cell value to int (so that they are integers).
    for column in number_edit_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')

    # Values conversion cells from false to 0, true to 1.
    columns_for_change_01 = ['fireplace', 'pool', 'furnished'] # Columns for change, leave , 'terrace' for now (not well maintained inimmoweb)
    df[columns_for_change_01] = df[columns_for_change_01].replace({False:0, True:1}).fillna(0).astype('Int64')

    # Converting the postal codes to dtype 'str'
    df['locality_code'] = df['locality_code'].astype(str)

    # Dropping a dutch postal code
    df = df[df['locality_code'] != '4524 JM']
    
    # Copy dataframe contents to a new dataframe, via removal of duplicates
    df = df.drop_duplicates(subset=['locality_latitude', 'locality_longitude', 'street', 'number', 'subtype'])

    return df

def subtype_consolidation(df):

    grp = 'subtype'         # select feature to consolidate
    steps = {}              # initiate dictionary of the different steps, to iterate over for the plots

    steps['before'] = df[grp].value_counts()         # Checks which subtypes and how many entries in the original dataset
                                                    # --> store as value of dictionary entry

    # Remove lines where 'Starting_price' is True
    """counter_has_starting_price = (df['starting_price'] == True).sum()
    df = df[df['starting_price'] != True]"""

    # Cleaning data from subtypes that are not our focus:
    subtypes_to_remove = ['House group', 'Chalet', 'Bungalow', 'Castle', 'Other property', 'Exceptional property']
    counter_subtypes_removed = (df['subtype'].isin(subtypes_to_remove)).sum()
    df = df[~df['subtype'].isin(subtypes_to_remove)]

    steps['after'] = df[grp].value_counts()         # Checks which subtypes and how many entries in the original dataset --> store as value of dictionary entry
    
    return df, steps

def missing_values(df):
    # Dealing with missing values: assumptions and operations on the dataset

    # Overview and grouping of the datacolumns for loop later on
    numerical_columns = ['price','net_habitable_surface','land_surface','facade_count','bedroom_count','garden_area', 'terrace_area']
    #garden_area', 'terrace_area' are not reliably maintained, just change <NA> to NaN to avoid errors 

    categorical_columns = ['kitchen_type','building_condition','epc'] # 'subtype' will be approached differently (see below)
    binary_columns = ['pool','fireplace','furnished'] # 'hasGarden', 'terrace', are not reliably maintained, leave as is so far

    # Dealing with missing values 
    # # Numerical columns: if NaN or 0, assigning the value to the median of the column (Imputation)
    for col in numerical_columns:
        # Calculate the median excluding NaN and 0 values
        col_median = df[col][df[col] != 0].dropna().median()
        # Replace 0 values with the median and fill NaN values
        df[col] = df[col].replace(0, col_median).fillna(col_median)
        # Replace None with the median
        df[col] = df[col].replace('None', 0).fillna(0)

    # # Binary columns: if NaN or 0, assigning the value to 0
    for col in binary_columns:
        # Replace Nan fill NaN values with 0
        df[col] = df[col].replace('Nan', 0).fillna(0)

    #df['kitchen_type'] = df['kitchen_type'].replace({0: np.nan, 'Nan': np.nan})

    # # Categorical columns: if NaN or 0, assigning the value to 0
    for col in categorical_columns:
        df[col] = df[col].replace({0: np.nan, 'Nan': np.nan})
        col_mode = df[col][df[col] != 0].dropna().mode()[0]
        df[col] = df[col].fillna(col_mode)

    return df

def unnecessary_columns(df, columns_to_drop):
    # Removing unnecessary columns
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

def adjust_datatypes(df):
    # transform 'int64' to 'Int64' to handle NaN

    int_columns = df.select_dtypes(include=['int64'])

    for col in int_columns.columns:
        df[col] = df[col].astype('Int64')

    # transform Net Habitable surface to Int64
    df['net_habitable_surface'] = df['net_habitable_surface'].astype('Int64')

    # transform dtypes 'object' to 'category'
    int_columns = df.select_dtypes(include=['object'])

    for col in int_columns.columns:
        df[col] = df[col].astype('category')

    df['street'] = df['street'].astype('str')
    df['number'] = df['number'].astype('str')

    df['locality_code'] = df['locality_code'].astype('category')

    df['fireplace'] = df['fireplace'].astype('bool')
    df['pool'] = df['pool'].astype('bool')
    df['terrace'] = df['terrace'].astype('bool')
    df['furnished'] = df['furnished'].astype('bool')

    return df

def mapping_subtypes(df, steps):
    # Lumping together subtypes via mapping
    grp = 'subtype'

    # Showing each subtype and its count, before mapping
    print("Before mapping: ", df[grp].value_counts())

    # Lumping together subtypes via mapping
    category_map = {'Mansion': 'Villa', 'Manor house': 'Villa', 'Country cottage':'Farmhouse','Town house':'House','Mixed use building':'Commercial','Apartment block':'Commercial'}
    df[grp] = df[grp].map(category_map).fillna(df[grp])

    # Showing each subtype and its count, after mapping
    print("After mapping:", df[grp].value_counts())
    
    steps['after mapping'] = df[grp].value_counts()         # Checks which subtypes and how many entries in the dataset
                                                    # --> store as value of dictionary entry
    return df, steps

def subtype_filter(df, steps):
    grp = 'subtype'

    # Filter data on subtype 'house' only
    df = df[df[grp] == 'House']
    
    steps['after filter'] = df[grp].value_counts()         # Checks which subtypes and how many entries in the original dataset
                                                    # --> store as value of dictionary entry
    return df, steps

def plot_cleanup(steps,grp):
 
    # Create a figure with 4 subplots
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    sns.color_palette("colorblind")

    # Flatten the ax array for easier indexing
    ax = ax.ravel()

    # Loop over the columns and create a scatter plot for each
    for index, (key, value) in enumerate(steps.items()):
        sns.barplot(x=value.values, y=value.index, orient="h", ax=ax[index])
        ax[index].set_title(f"Consolidation on {grp} - {key}") # Set title for each plot

    #plt.legend(loc='upper center')     # Move the legend to the right side
    plt.tight_layout()
    plt.show()

    return
