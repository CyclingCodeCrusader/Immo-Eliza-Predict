import pandas as pd
from tkinter import filedialog
from utils.utils import open_csv_as_dataframe, create_pkl_from_df, create_df_from_pkl, barplot
from functions.preprocessing_functions import map_province, assign_city_based_on_proximity_multiple_radii, outlier_handling_numerical, outlier_handling_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from utils.utils import figure_barplots, figure_boxplots, barplot


class Preprocessor:
    """
    Create a class `Preprocessor` that contains these attributes:
    - `num_data_cols` which is a list of numerical data columns 
    - `cat_data_cols` which is a list of numerical data columns
    - `category_maps` which is a dictionary of category maps that are dictionaries of category: category type

    And some methods:
    - `outlier_handling_numerical(num_data_cols)` that will check the best outlier handling method for numerical values (IQR or Z method) and remove outlier.
    - `outlier_handling_categorical(cat_data_cols)` that will check for the rare values and map them to other field values.
    - `outliers_Z(df, num_data_col)` that will determine outliers using the Z method.
    - `outliers_IQR(df, num_data_col)` that will determine outliers using the IQR method.
    - `store(filename)` store the repartition in an excel file
    - __str__ dunder method
    - __init__ constructor
    """

    def __init__(self):
        self.input_filepath = r'data\after_datacleaning.pkl'
        self.num_data_cols = ['price','net_habitable_surface','bedroom_count','facade_count'] #something wrong with 'land_surface', it gives a 0 when calcuating skew
        self.cat_data_cols = {'kitchen_type': {'Usa hyper equipped': 'Hyper equipped', 'Usa semi equipped': 'Semi equipped', 'Usa uninstalled':'Not installed', 'Usa installed':'Installed'}, 
                              'building_condition': {'To restore': 'To renovate'},
                              'epc': {'A+': 'A', 'A++': 'A', 'G':'F'}}

    def figure_barplots(self, steps,col):
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Function to create plots
        sns.color_palette("colorblind")

        # Create a figure with the number of required subplots and grid distribution
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
        
        # Flatten the ax array for easier indexing
        ax = ax.ravel()

        # Loop over the columns and create a scatter plot for each
        for index, (key, value) in enumerate(steps.items()):
            sns.barplot(x=value.index, y=value.values, ax=ax[index])
            ax[index].set_title(f"Bar plot consolidation categorical data {col} - {key}") # Set title for each plot

        #plt.legend(loc='upper center')     # Move the legend to the right side
        plt.tight_layout()
        plt.show()

        return

    def figure_boxplots(self, sets):
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Function to create plots
        sns.color_palette("colorblind")

        # Create a figure with the number of required subplots and grid distribution
        fig, ax = plt.subplots(nrows=len(sets) // 4 if len(sets) % 4 == 0 else len(sets) // 4 + 1, ncols=len(sets), figsize=(15, 5))
        
        # Flatten the ax array for easier indexing
        ax = ax.ravel()

        # Loop over the columns and create a scatter plot for each
        for i, set in enumerate(sets):
            sns.boxplot(x=set, orient='h', color='blue', ax=ax[i])
            ax[i].set_title(f'{set.name} - box plot for outlier detection') # Set title for each plot

        #plt.legend(loc='upper center')     # Move the legend to the right side
        plt.tight_layout()
        plt.show()
        
        return

    def outliers_Z(self, df, num_data_col):
        # This function does detection and removal of outliers via the Z method
        upper_limit = df[num_data_col].mean() + 4*df[num_data_col].std()
        lower_limit = df[num_data_col].mean() - 4*df[num_data_col].std()

        print('upper limit:', upper_limit)
        print('lower limit:', lower_limit)

        # find the outlier

        df.loc[(df[num_data_col] > upper_limit) | (df[num_data_col] < lower_limit)]

        # remove outliers

        df_post_Z = df.loc[(df[num_data_col] < upper_limit) & (df[num_data_col] > lower_limit)]
        print('before removing outliers:', len(df))
        print('after removing outliers:', len(df_post_Z))
        print('outliers:', len(df) - len(df_post_Z))

        return df_post_Z

    def outliers_IQR(self, df, num_data_col):
        # This function does detection and removal of outliers via the IQR method
        # In this method, we determine quartile values ​​Q1 (25th percentile) and Q3 (75th percentile) and then cal
        # Outliers are those that fall outside the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]

        q1 = df[num_data_col].quantile(0.25)
        q3 = df[num_data_col].quantile(0.75)
        iqr = q3 - q1

        # Specifying the scope of outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Data filtering
        df_post_IQR = df[(df[num_data_col] >= lower_bound) & (df[num_data_col] <= upper_bound)]

        return df_post_IQR

    def outlier_handling_numerical(self):
        
        for col in self.num_data_cols:        
            skew_before = self.df[col].skew()

            df_post_IQR = self.outliers_IQR(self.df, col)
            skew_after_IQR = df_post_IQR[col].skew()

            df_post_Z = self.outliers_Z(self.df, col)
            skew_after_Z = df_post_Z[col].skew()
            
            print("Skew before: ", skew_before, type(skew_before))

            if skew_after_IQR < skew_after_Z and skew_after_IQR <= skew_before:
                print("Skew lowest after IQR method:")
                self.outliers_stats_plots(self.df, df_post_IQR, col)
                self.df = df_post_IQR

            elif skew_after_Z < skew_after_IQR and skew_after_Z <= skew_before:
                print("Skew lowest after Z method:")
                self.outliers_stats_plots(self.df, df_post_Z, col)
                self.df = df_post_Z

            else:
                print("No effect of outlier methods:")
                self.outliers_stats_plots(self.df, self.df, col)
                self.df = df

        return self.df

    def outliers_stats_plots(self, df, df_post, col):
        
        print("Before outlier handling: ")

        print(df[col].agg(['count','skew','mean','median']))
        print(df[col].mode())

        print("After outlier handling: ")
        print(df_post[col].agg(['count','skew','mean','median']))
        print(df_post[col].mode())

        sets = [df[col],df_post[col]]

        self.figure_boxplots(sets)

        return

    def outlier_handling_categorical(self):

        for key, value in self.cat_data_cols.items():    
            steps = {}

            steps['before'] = self.df[key].value_counts()

            # Determining the rare values (threshold 5% of the dataset)
            threshold = 0.05 * len(self.df)  
            rare_categories = steps['before'][steps['before'] < threshold]
            #print("Rare Values:", rare_categories)

            # Assign the rare value to another value
            
            self.df[key] = self.df[key].map(value).fillna(self.df[key])

            steps['after'] = self.df[key].value_counts()
            
            self.figure_barplots(steps,key)

        return self.df
  
    def run_workflow(self):
        self.df = create_df_from_pkl(default_path = r'data\cleaned.pkl')
        self.outlier_handling_numerical()
        self.outlier_handling_categorical()
        print(type(self.df))
        print(self.df.info())
        create_pkl_from_df(self.df, file_path = r'data\preprocessed.pkl') # Give dataframe to save, and path to file
        return self.df

"""# Part on outlier handling of categorical columns, via IQR or Z method and removing of outlier samples
df11 = outlier_handling_numerical(df10, num_data_cols=['price','net_habitable_surface','bedroom_count','facade_count']) #something wrong with 'land_surface', it gives a 0 when calcuating skew

# Part on outlier handling of categorical columns, via assigning rare values to other categories
df12 = outlier_handling_categorical(df11, cat_data_col='kitchen_type', category_map = {'Usa hyper equipped': 'Hyper equipped', 'Usa semi equipped': 'Semi equipped', 'Usa uninstalled':'Not installed', 'Usa installed':'Installed'})
df13 = outlier_handling_categorical(df12, cat_data_col='building_condition', category_map = {'To restore': 'To renovate'})
df14 = outlier_handling_categorical(df13, cat_data_col='epc', category_map = {'A+': 'A', 'A++': 'A', 'G':'F'})

save_dataframe_to_pkl(df14, file_path = r'data\after_analysis.pkl') # Give dataframe to save, and path to file
"""