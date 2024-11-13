import pickle
from functions.utils import create_df_from_pkl
from functions.train_model_functions import ordinal_encoding, OneHot_encoding
from functions.train_model_functions import models_linear, models_polynomial, models_treebased, create_Xy, polynomial_simple, XGBoost
from functions.train_model_functions import save_best_model, load_prediction_model
from functions.utils import open_csv_as_dataframe, create_pkl_from_df, create_df_from_pkl, barplot
from functions.preprocessing_functions import map_province, assign_city_based_on_proximity_multiple_radii, outlier_handling_numerical, outlier_handling_categorical
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
#from catboost import CatBoostRegressor, Pool
#from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from  geopy.geocoders import Nominatim

from functions.utils import figure_barplots, figure_boxplots, barplot

class Processor():
    """
    Create a class `Processor` to that contains attributes and methods to be performed on the training dataset and the new data(set).
    The class contains the following attributes:
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
    
        self.cities_data = {
                        'city': ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liège', 'Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas'],
                        'locality_latitude': [50.8503, 51.2211, 51.0543, 51.2093, 50.6050, 50.4674, 50.8798, 50.4542, 50.9403, 51.1449],
                        'locality_longitude': [4.3517, 4.4120, 3.7174, 3.2240, 5.5797, 4.8712, 4.7033, 3.9514, 4.0364, 4.1525]
                            }
        self.radius_list = [5, 10, 15]
        # Overview and grouping of the datacolumns for loop later on
        self.numerical_columns = ['bedroom_count', 'net_habitable_surface', 'facade_count','land_surface']
        self.numerical_columns_backlog = ['terrace_surface','garden_surface']

        self.unnecessary_columns = ['id','locality_name','street', 'number','locality_latitude','locality_longitude']
        self.derivative_columns = ['price_per_sqm','price_per_sqm_land']

        self.categorical_columns = ['subtype','kitchen_type','building_condition','epc','locality_code','province', 'assigned_city','assigned_city_5', 'assigned_city_10', 'assigned_city_15']

        self.binary_columns = ['pool', 'fireplace','furnished', 'has_assigned_city','has_assigned_city_5', 'has_assigned_city_10', 'has_assigned_city_15'] # 'hasTerrace', not reliably maintained so leaving it out of analyzing/visualization

        self.ordinal_encoded_columns = []
        #self.ordinal_encoded_columns = ['kitchen_type_encoded', 'building_condition_encoded','epc_encoded']
        self.onehot_columns = ['province']
        self.ordinal_encode_columns = ['kitchen_type','building_condition','epc']

        self.target_column = ['price']
        self.predictor_columns = self.numerical_columns + self.ordinal_encode_columns + ['has_assigned_city_10'] + ['province']

        self.predict_columns_payload = ['bedroom_count', 'net_habitable_surface', 'facade_count', 'land_surface', 
                        'kitchen_type_ord_enc', 'building_condition_ord_enc', 'epc_ord_enc', 'locality_code'] #'has_assigned_city_10', 
        
    def map_province(self, locality_code):
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

    def get_province(self):    
        # Creating the column province
        self.df['province'] = self.df['locality_code'].apply(self.map_province)

        # Assigning the dtypes
        self.df['province'] = self.df['province'].astype('category')
        
        #barplot(self.df, feature='province')                                          # call barplot function in utils. plots a barplot of the valuecounts in province
        
        return self.df

    def assign_city_based_on_proximity_multiple_radii(self):
        
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
        
        cities_df = pd.DataFrame(self.cities_data)
        cities_gdf = gpd.GeoDataFrame(cities_df, geometry=gpd.points_from_xy(cities_df.locality_longitude, cities_df.locality_latitude))
        
        cities_gdf = cities_gdf.set_geometry('geometry')
        
        house_geo = pd.DataFrame(self.df[['id', 'locality_latitude', 'locality_longitude']])
        house_geo_gdf = gpd.GeoDataFrame(house_geo, geometry=gpd.points_from_xy(house_geo.locality_longitude, house_geo.locality_latitude))

        house_geo_gdf = house_geo_gdf.set_geometry('geometry')
        
        for radius in self.radius_list:
            
            cities_gdf['buffer'] = cities_gdf.geometry.buffer(radius / 111)
            
            cities_gdf = cities_gdf.set_geometry('buffer')  
            
            joined_gdf = gpd.sjoin(house_geo_gdf, cities_gdf[['city', 'buffer']], how='left', predicate='intersects')
            joined_gdf = joined_gdf.drop_duplicates(subset='id')
            
            house_geo_gdf[f'assigned_city_{radius}'] = joined_gdf['city']
            
            self.df = pd.merge(self.df, house_geo_gdf[['id', f'assigned_city_{radius}']], on='id', how='left')
            
            self.df[f'has_assigned_city_{radius}'] = self.df[f'assigned_city_{radius}'].notna()
            self.df[f'has_assigned_city_{radius}'] = self.df[f'has_assigned_city_{radius}'].astype('bool')

        return self.df

    def get_geolocation(self):
        # Creating the columns longitude and longitude
        self.df['locality_latitude'] = self.df['locality_code'].apply(self.zip_to_geoloc_lat)
        self.df['locality_longitude'] = self.df['locality_code'].apply(self.zip_to_geoloc_long)
        return self.df

    def zip_to_geoloc_lat(self, locality_code):
        # Initialize Nominatim API
        geolocator = Nominatim(user_agent="zip_to_geolocation")
        
        # Get location based on ZIP code
        location = geolocator.geocode({"postalcode": int(locality_code), "country": "Belgium"})
        
        if location:
            return location.latitude
        else:
            return None

    def zip_to_geoloc_long(self, locality_code):
        # Initialize Nominatim API
        geolocator = Nominatim(user_agent="zip_to_geolocation")
        
        # Get location based on ZIP code
        location = geolocator.geocode({"postalcode": int(locality_code), "country": "Belgium"})
        
        if location:
            return location.longitude
        else:
            return None

    def zip_to_geolocation(self, locality_code):
        # Initialize Nominatim API
        geolocator = Nominatim(user_agent="zip_to_geolocation")
        
        # Get location based on ZIP code
        location = geolocator.geocode({"postalcode": locality_code})
        
        if location:
            return location.latitude, location.longitude
        else:
            return None, None

    def create_df_from_predict_input(self, test_data):
        #Create a new house 
        new_house = np.array(test_data)
        self.df= pd.DataFrame(new_house, columns=self.predict_columns_payload)
        return self.df

    def train_workflow(self):
        self.df = create_df_from_pkl(default_path = r'data\preprocessed.pkl') # Fill your path to file
        self.get_province()
        self.assign_city_based_on_proximity_multiple_radii()
        create_pkl_from_df(self.df, file_path = r'data\processed.pkl') # Give dataframe to save, and path to file
        return self.df

    def predict_workflow(self, test_data):
        self.df = self.create_df_from_predict_input(test_data)
        self.get_geolocation()
        self.df['locality_code'] = self.df['locality_code'].astype(str)     # Convert the 'locality_code' column to string type, to function in the get province and map province methods
        self.df['id'] = 1                                                   # Add a column 'id', with a value 1, so that the dataframe can run through the following methods
        self.get_province()
        self.assign_city_based_on_proximity_multiple_radii()
        print("self.df at end of predict workflow:", type(self.df))
        print(self.df.columns)
        #print(self.df.info())
        return self.df