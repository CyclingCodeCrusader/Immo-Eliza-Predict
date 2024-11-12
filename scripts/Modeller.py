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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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