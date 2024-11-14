import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from utils.utils import figure_barplots, figure_boxplots, barplot

def map_province(locality_code):
    if locality_code.startswith('1'):
        return 'Brussels' if int(locality_code) < 1300 else 'Brabant_Wallon'
    elif locality_code.startswith('2'):
        return 'Antwerp'
    elif locality_code.startswith('4'):
        return 'Liege'
    elif locality_code.startswith('5'):
        return 'Namur'
    elif locality_code.startswith('6'):
        return 'Luxembourg'
    elif locality_code.startswith('7'):
        return 'Hainaut'
    elif locality_code.startswith('8'):
        return 'West_Flanders'
    elif locality_code.startswith('9'):
        return 'East_Flanders'
    elif locality_code.startswith('3'):
        return 'Flemish_Brabant' if int(locality_code) < 3500 else 'Limburg'
    else:
        return None 

def get_province(df):    
    # Creating the column province
    df['province'] = df['locality_code'].apply(map_province)

    # Assigning the dtypes
    df['province'] = df['province'].astype('category')
    
    return df

def assign_city_based_on_proximity(df, cities_data):
    # Creating a column for proximity to the 10 main Belgian cities:
    cities_data = {'city': ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liege','Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas'],
                'locality_latitude': [50.8503, 51.2211, 51.0543, 51.2093, 50.6050, 50.4674, 50.8798, 50.4542, 50.9403, 51.1449],
                'locality_longitude': [4.3517, 4.4120, 3.7174, 3.2240, 5.5797, 4.8712, 4.7033, 3.9514, 4.0364, 4.1525],
                'radius': [10 for x in range(10)]}

    cities_df = pd.DataFrame(cities_data)

    # Make a geodataframe from the cities dataframe
    cities_gdf = gpd.GeoDataFrame(cities_df,geometry=gpd.points_from_xy(cities_df.locality_longitude, cities_df.locality_latitude))

    # Creating the buffer/radius zone (set on 10km)
    cities_gdf['buffer'] = cities_gdf.geometry.buffer(cities_gdf['radius'] / 111)
    cities_gdf = cities_gdf.set_geometry('buffer')

    # Checking and slicing original data and creating a new dataframe house_geo
    house_geo= pd.DataFrame(df[['id', 'locality_latitude', 'locality_longitude']]).copy()

    # Making a geo dataframe from the dataframe
    house_geo_gdf = gpd.GeoDataFrame(house_geo,geometry=gpd.points_from_xy(house_geo.locality_longitude, house_geo.locality_latitude))

    # Join the two gdf geodataframes 'house_geo_gdf' and 'cities_gdf'
    joined_gdf = gpd.sjoin(house_geo_gdf, cities_gdf[['city', 'buffer']], how='left', predicate='intersects', lsuffix='_house', rsuffix='_city')
    house_geo_gdf['assigned_city'] = joined_gdf['city']

    # Merge the assigned city column to the main dataframe
    df = pd.merge(df, house_geo_gdf[['id', 'assigned_city']], on='id', how='left')

    #Make a boolean column of Assigned_City and transform to bool
    df['has_assigned_city'] = df['assigned_city'].isnull()
    df['has_assigned_city'] = df['has_assigned_city'].astype('bool')

    return df

def assign_city_based_on_proximity_multiple_radii(df, cities_data, radius_list):
    
    """
    This function assigns cities to each row in the dataframe based on proximity 
    to the 10 main Belgian cities within multiple given distance radii.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing house data with 'locality_latitude' and 'locality_longitude' columns.
    cities_data (dict): Dictionary containing city data ('City', 'locality_latitude', 'locality_longitude').
    radius_list (list): List of radii (in kilometers) to consider for proximity.
    
    Returns:
    df (pd.DataFrame): Updated DataFrame with 'Assigned_City' columns and proximity-based boolean columns.
    """
    
    cities_df = pd.DataFrame(cities_data)
    cities_gdf = gpd.GeoDataFrame(cities_df, geometry=gpd.points_from_xy(cities_df.locality_longitude, cities_df.locality_latitude))
    
    cities_gdf = cities_gdf.set_geometry('geometry')
    
    house_geo = pd.DataFrame(df[['id', 'locality_latitude', 'locality_longitude']])
    house_geo_gdf = gpd.GeoDataFrame(house_geo, geometry=gpd.points_from_xy(house_geo.locality_longitude, house_geo.locality_latitude))

    house_geo_gdf = house_geo_gdf.set_geometry('geometry')
    
    for radius in radius_list:
        
        cities_gdf['buffer'] = cities_gdf.geometry.buffer(radius / 111)
        
        cities_gdf = cities_gdf.set_geometry('buffer')  
        
        
        joined_gdf = gpd.sjoin(house_geo_gdf, cities_gdf[['city', 'buffer']], how='left', predicate='intersects')
        joined_gdf = joined_gdf.drop_duplicates(subset='id')
        
        house_geo_gdf[f'assigned_city_{radius}'] = joined_gdf['city']
        
        
        df = pd.merge(df, house_geo_gdf[['id', f'assigned_city_{radius}']], on='id', how='left')
        
        df[f'has_assigned_city_{radius}'] = df[f'assigned_city_{radius}'].notna()
        df[f'has_assigned_city_{radius}'] = df[f'has_assigned_city_{radius}'].astype('bool')
    
    return df

def outliers_Z(df, num_data_col):
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

def outliers_IQR(df, num_data_col):
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

def outlier_handling_numerical(df, num_data_cols):
    
    for col in num_data_cols:        
        skew_before = df[col].skew()

        df_post_IQR = outliers_IQR(df, col)
        skew_after_IQR = df_post_IQR[col].skew()

        df_post_Z = outliers_Z(df, col)
        skew_after_Z = df_post_Z[col].skew()
        
        print("Skew before: ", skew_before, type(skew_before))

        if skew_after_IQR < skew_after_Z and skew_after_IQR <= skew_before:
            print("Skew lowest after IQR method:")
            outliers_stats_plots(df, df_post_IQR, col)
            df = df_post_IQR

        elif skew_after_Z < skew_after_IQR and skew_after_Z <= skew_before:
            print("Skew lowest after Z method:")
            outliers_stats_plots(df, df_post_Z, col)
            df = df_post_Z

        else:
            print("No effect of outlier methods:")
            outliers_stats_plots(df, df, col)
            df = df

    return df

def outliers_stats_plots(df, df_post, col):
    
    print("Before outlier handling: ")

    print(df[col].agg(['count','skew','mean','median']))
    print(df[col].mode())

    print("After outlier handling: ")
    print(df_post[col].agg(['count','skew','mean','median']))
    print(df_post[col].mode())

    sets = [df[col],df_post[col]]

    figure_boxplots(sets)

    return

def outlier_handling_categorical(df, cat_data_col, category_map):
        
    steps = {}

    steps['before'] = df[cat_data_col].value_counts()

    # Determining the rare values (threshold 5% of the dataset)
    threshold = 0.05 * len(df)  
    rare_categories = steps['before'][steps['before'] < threshold]
    #print("Rare Values:", rare_categories)

    # Assign the rare value to another value
    
    df[cat_data_col] = df[cat_data_col].map(category_map).fillna(df[cat_data_col])

    steps['after'] = df[cat_data_col].value_counts()
    
    figure_barplots(steps,cat_data_col)

    return df
