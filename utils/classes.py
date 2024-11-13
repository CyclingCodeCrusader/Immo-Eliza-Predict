import pandas as pd
from tkinter import filedialog
from functions.utils import open_csv_as_dataframe, create_pkl_from_df, create_df_from_pkl, barplot
from functions.preprocessing_functions import map_province, assign_city_based_on_proximity_multiple_radii, outlier_handling_numerical, outlier_handling_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from functions.utils import figure_barplots, figure_boxplots, barplot


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

import pickle
import pandas as pd
import joblib

from functions.utils import create_df_from_pkl
from functions.train_model_functions import ordinal_encoding, OneHot_encoding
from functions.train_model_functions import models_linear, models_polynomial, models_treebased, create_Xy, polynomial_simple, XGBoost
from functions.train_model_functions import save_best_model, load_prediction_model
from functions.utils import open_csv_as_dataframe, create_pkl_from_df, create_df_from_pkl, barplot
from functions.preprocessing_functions import map_province, assign_city_based_on_proximity_multiple_radii, outlier_handling_numerical, outlier_handling_categorical
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, PolynomialFeatures
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
        
        barplot(self.df, feature='province')                                          # call barplot function in utils. plots a barplot of the valuecounts in province
        
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

    def create_Xy(self):
        """ Save the target column in the variable y, and the predictor columns in the variable X
            Create feature matrix X and target matrix y. This is done here, 
            because it is easier to perform label encoding and onehot encoding on the feature matrix instead of the full dataframe, 
            so that you don't have to include all the newly generated encoded columns """
        X = self.df[self.predictor_columns]
        y = self.df[self.target_column]

        return X, y

    def ordinal_encoding(self, X):
        #This categorical data has a natural order we encode it in a way that reflects this ordering. We will use ordinal Encoding.
        
        # Define the custom order for the 'Kitchen_type' column
        ordinals_kitchen = [['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']]  # Order for each ordinal column
        ordinals_building_condition = [['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']]  # Order for each ordinal column
        ordinals_epc = [['F', 'E', 'D', 'C', 'B', 'A']]  # Order for each ordinal column

        ordinals_list = [ordinals_kitchen, ordinals_building_condition, ordinals_epc]
        #print(type(X))

        for i, col in enumerate(self.ordinal_encode_columns):
            # Initialize OrdinalEncoder with the specified categories
            encoder = OrdinalEncoder(categories=ordinals_list[i])
            name_ord_enc = f"{col}_ord_enc"
            self.ordinal_encoded_columns.append(name_ord_enc)
            # Fit and transform the column
            #print(col, type(X[[col]]))
            #X[name_ord_enc] = encoder.fit_transform(X[[col]]) # syntax from solution of error message dfmi.loc[:, ('one', 'second')]

            X = X.assign(**{name_ord_enc: encoder.fit_transform(X[[col]])})

            #f"{col}_ord_enc" name_ord_enc
            X = X.drop(columns = col)

        return X

    def onehot_encoding(self, X):
        
        for col in self.onehot_columns:
            # One-hot encode in the dataframe
            X = pd.get_dummies(X, columns=[col], drop_first=True)

        return X

    def run_workflow(self):
        self.df = create_df_from_pkl(default_path = r'data\preprocessed.pkl') # Fill your path to file
        self.get_province()
        self.assign_city_based_on_proximity_multiple_radii()
        X,y = self.create_Xy()
        X = self.ordinal_encoding(X)
        X = self.onehot_encoding(X)
        print(type(self.df))
        print(self.df.info())
        create_pkl_from_df(X, file_path = r'data\processed_X.pkl') # Give dataframe to save, and path to file
        create_pkl_from_df(y, file_path = r'data\processed_y.pkl') # Give dataframe to save, and path to file
        return X,y

import pickle
import pandas as pd
import joblib

from functions.utils import create_df_from_pkl
from functions.train_model_functions import ordinal_encoding, OneHot_encoding
from functions.train_model_functions import models_linear, models_polynomial, models_treebased, create_Xy, polynomial_simple, XGBoost
from functions.train_model_functions import save_best_model, load_prediction_model
from functions.utils import open_csv_as_dataframe, create_pkl_from_df, create_df_from_pkl, barplot
from functions.preprocessing_functions import map_province, assign_city_based_on_proximity_multiple_radii, outlier_handling_numerical, outlier_handling_categorical
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
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

class Modeller():
    """
    Create a class `Modeller` to that contains attributes and methods to be performed on the training dataset and the new data(set).
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

    def __init__(self, X, y):
        self.X = X
        self.y = y
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
        self.to_encode_columns = ['kitchen_type','building_condition','epc']

        self.target_column = ['price']
        self.predictor_columns = self.numerical_columns + self.to_encode_columns + ['has_assigned_city_10'] # + ['province']

    def best_pipeline(self):        
        #self.encoder = ColumnTransformer(
        #        transformers=[('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
        
        # Create a pipeline that includes the scaler, the encoder and a placeholder for the model
        self.pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()),
                #('encoder', self.encoder),
                ('regressor', None)  # Placeholder for the model
                ])

        # Define the models and parameters to explore
        self.param_grid = [
            #{'regressor': [LinearRegression()]},                    # No hyperparameters for LinearRegression
            #{'regressor': [Ridge()],
            #'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000],
            #'regressor__solver': ["auto", "cholesky","sparse_cg"]}, #"lbfgs",
            #{'regressor': [Lasso()],
            #'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000]},
            {'regressor': [ElasticNet()],
            'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
            ]

        # Define KFold cross-validation strategy
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Set up GridSearchCV with the pipeline, parameter grid, and KFold cross-validation
        grid_search = GridSearchCV(estimator = self.pipeline, param_grid = self.param_grid, cv=kf, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Retrieve the best model and parameters
        print("PERFORMANCE OF LINEAR REGRESSION MODELS: \n ----LinearRegression \n ----regularization: Ridge, Lasso, ElasticNet")

        # Retrieve the best pipeline
        self.best_model = grid_search.best_estimator_

        self.best_params = grid_search.best_params_
        
        print("Best model:", self.best_model)
        print("Best parameters:", self.best_params)

        print("Best model score on training set: ", self.best_model.score(X_train, y_train))
        print("Best model score on test set: ",self.best_model.score(X_test, y_test))
        
        #  Get the feature names generated after encoding
        self.feature_names = self.pipeline.named_steps['encoder'].get_feature_names_out()
        print("Feature names after encoding:", self.feature_names)
        
        # Predict on test data using the best model
        y_pred = self.best_model.predict(X_test)
        
        # Evaluate the model
        self.evaluation(y_test.to_numpy(), y_pred)
        print(X_train)

        return self.best_model

    def models_linear(self):        
        # Create the pipeline with a placeholder for the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', None)  # Placeholder for the model
        ])

        # Define the models and parameters to explore
        param_grid = [
            #{'regressor': [LinearRegression()]},                    # No hyperparameters for LinearRegression
            #{'regressor': [Ridge()],
            #'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000],
            #'regressor__solver': ["auto", "cholesky","sparse_cg"]}, #"lbfgs",
            #{'regressor': [Lasso()],
            #'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000]},
            {'regressor': [ElasticNet()],
            'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
            ]

        # Define KFold cross-validation strategy
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Set up GridSearchCV with the pipeline, parameter grid, and KFold cross-validation
        grid_search = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv=kf, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Retrieve the best model and parameters
        print("PERFORMANCE OF LINEAR REGRESSION MODELS: \n ----LinearRegression \n ----regularization: Ridge, Lasso, ElasticNet")

        # Retrieve the best pipeline
        best_pipeline = grid_search.best_estimator_

        best_params = grid_search.best_params_
        
        print("Best model:", best_pipeline)
        print("Best parameters:", best_params)

        print("Best model score on training set: ", best_pipeline.score(X_train, y_train))
        print("Best model score on test set: ",best_pipeline.score(X_test, y_test))

        # Predict on test data using the best model
        y_pred = best_pipeline.predict(X_test)
        
        # Evaluate the model
        self.evaluation(y_test.to_numpy(), y_pred)
        print(X_train)

        return best_pipeline

    def evaluation(self, y_test, y_pred):

        import numpy as np

        """Calculates the Mean Signed Error.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            RMSE, MAE, Mean Signed Error.
        """
        # Calculate Root Mean Squared Error
        rmse = round(root_mean_squared_error(y_test, y_pred), 1)
        print("Root Mean Squared Error of predictions vs test data:", rmse)
        
        # Calculate Mean Absolute Error
        mae = round(mean_absolute_error(y_test, y_pred),1)
        print("Mean Absolute Error for test (MAE):", mae)
        
        # Calculate Mean Signed Error
        mse = round(np.mean(y_pred - y_test),1)
        print("Mean Signed Error for test (MSE):", mse)

        return

    def run_workflow(self):
        
        return

"""#part of processing of datasets (both during training and prediction) prior to modeling
df = map_province(df)
#barplot(df,feature='province')

df = assign_city_based_on_proximity_multiple_radii(df, cities_data, radius_list)

# Part of model training pipeline starts here starts here
#scaler, encoder onehot, encoder ordinal, regressor go in pipeline to save to joblib file
ordinal_encoding
OneHot_encoding
models_linear
pipeline
joblib 

# Save our target column in the variable y, and the predictor columns in the variable X
target_column = ['price']
predictor_columns = numerical_columns + to_encode_columns + ['has_assigned_city_10'] #['province'] + 
# Create feature matrix X and target matrix y. This is done here, 
# because it is easier to perform label encoding and onehot encoding on the feature matrix instaed of the full dataframe, 
# so that you don't have to include all the newly generated encoded columns 
X, y = create_Xy(df1, predictor_columns, target_column)

# Part on label encoding
X = ordinal_encoding(X, to_encode_columns)

# Part on one hot encoding
#X = OneHot_encoding(X, columns = ['province'])

print(X.info())

best_pipeline = models_linear(X,y)
#best_model, best_params = models_polynomial(X,y)
#polynomial_simple(X,y)
#models_treebased(X,y)
#XGBoost(X,y)

# Save the best pipeline with joblib
joblib.dump(best_pipeline, 'best_model_pipeline.joblib')"""