import numpy as np
import pandas as pd
from joblib import load, dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
#from sklearn.ensemble import RandomForestRegressor 
#from sklearn.tree import DecisionTreeRegressor
#from catboost import CatBoostRegressor, Pool
#from xgboost import XGBRegressor
#from sklearn.svm import SVR

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

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

    def __init__(self, df):
        self.df = df
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

        self.onehot_columns = ['province']
        self.ordinal_encode_columns = ['kitchen_type','building_condition','epc']
        self.ordinal_encoded_columns = []
        #self.ordinal_encoded_columns = ['kitchen_type_encoded', 'building_condition_encoded','epc_encoded']

        self.target_column = ['price']
        self.predictor_columns = self.numerical_columns + self.ordinal_encode_columns + ['has_assigned_city_10'] + ['province']

        ['bedroom_count', 'net_habitable_surface', 'facade_count','land_surface', 'kitchen_type_ord_enc', 'building_condition_ord_enc',
       'epc_ord_enc', 'locality_code', 'locality_latitude',
       'locality_longitude', 'id', 'province', 'assigned_city_5',
       'has_assigned_city_5', 'assigned_city_10', 'has_assigned_city_10',
       'assigned_city_15', 'has_assigned_city_15']

        self.province_columns = ['province_Brabant_Wallon', 'province_Brussels', 'province_East_Flanders', 'province_Flemish_Brabant', 'province_Hainaut', 'province_Liege', 'province_Limburg', 'province_Luxembourg', 'province_Namur', 'province_West_Flanders']
       
        self.province_to_onehot = {'Brabant_Wallon': [1,0,0,0,0,0,0,0,0,0],
        'Brussels': [0,1,0,0,0,0,0,0,0,0], 
        'East_Flanders': [0,0,1,0,0,0,0,0,0,0], 
        'Flemish_Brabant': [0,0,0,1,0,0,0,0,0,0], 
        'Hainaut': [0,0,0,0,1,0,0,0,0,0], 
        'Liege': [0,0,0,0,0,1,0,0,0,0], 
        'Limburg': [0,0,0,0,0,0,1,0,0,0], 
        'Luxembourg': [0,0,0,0,0,0,0,1,0,0], 
        'Namur': [0,0,0,0,0,0,0,0,1,0], 
        'West_Flanders': [0,0,0,0,0,0,0,0,0,1]}

    def create_Xy(self):
        """ Save the target column in the variable y, and the predictor columns in the variable X
            Create feature matrix X and target matrix y. This is done here, 
            because it is easier to perform label encoding and onehot encoding on the feature matrix instead of the full dataframe, 
            so that you don't have to include all the newly generated encoded columns """
        self.X = self.df[self.predictor_columns]
        self.y = self.df[self.target_column]

        return self.X, self.y

    def df_fit_to_model(self):
        #Method to remove superfluous columns from the dataframe, so that only the feature columns, seen at fit remain
        self.X = self.df[self.predictor_columns]                   # fit the input df to the df used for model fitting
        #self.df = self.df.drop(columns='locality_code')

        return self.X

    def ordinal_encoding(self):
        #This categorical data has a natural order we encode it in a way that reflects this ordering. We will use ordinal Encoding.
        
        # Define the custom order for the 'Kitchen_type' column
        ordinals_kitchen = [['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']]  # Order for each ordinal column
        ordinals_building_condition = [['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']]  # Order for each ordinal column
        ordinals_epc = [['F', 'E', 'D', 'C', 'B', 'A']]  # Order for each ordinal column

        ordinals_list = [ordinals_kitchen, ordinals_building_condition, ordinals_epc]
        print(type(self.X))

        for i, col in enumerate(self.ordinal_encode_columns):
            # Initialize OrdinalEncoder with the specified categories
            encoder = OrdinalEncoder(categories=ordinals_list[i])
            name_ord_enc = f"{col}_ord_enc"
            self.ordinal_encoded_columns.append(name_ord_enc)
            # Fit and transform the column
            #print(col, type(X[[col]]))
            #X[name_ord_enc] = encoder.fit_transform(X[[col]]) # syntax from solution of error message dfmi.loc[:, ('one', 'second')]

            self.X = self.X.assign(**{name_ord_enc: encoder.fit_transform(self.X[[col]])})

            #f"{col}_ord_enc" name_ord_enc
            self.X = self.X.drop(columns = col)

        return self.X, self.ordinal_encoded_columns

    def onehot_encoding(self):
        
        for col in self.onehot_columns:
            # One-hot encode in the dataframe
            self.X = pd.get_dummies(self.X, columns=[col], drop_first=True)

        return self.X

    def best_linreg_model_pipeline(self, X,y):        
        self.X = X
        self.y = y
        
        # Create the pipeline with a placeholder for the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', None)  # Placeholder for the model
        ])

        # Define the models and parameters to explore
        param_grid = [
            {'regressor': [LinearRegression()]},                    # No hyperparameters for LinearRegression
            {'regressor': [Ridge()],
            'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000],
            'regressor__solver': ["auto", "cholesky","sparse_cg"]}, #"lbfgs",
            {'regressor': [Lasso()],
            'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000]},
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
        self.best_linreg_pipeline = grid_search.best_estimator_

        self.best_linreg_params = grid_search.best_params_
        
        print("Best model:", self.best_linreg_pipeline)
        print("Best parameters:", self.best_linreg_params)

        print("Best model score on training set: ", self.best_linreg_pipeline.score(X_train, y_train))
        print("Best model score on test set: ", self.best_linreg_pipeline.score(X_test, y_test))

        # Predict on test data using the best model
        y_pred = self.best_linreg_pipeline.predict(X_test)
        
        # Evaluate the model
        self.evaluation(y_test.to_numpy(), y_pred)
        
        
        print("X train uit best linreg model pipeline method: ", type(X_train))

        # printen van wat we hebben hier
        self.fitted_feature_columns = X_train.columns.tolist()
        print("De feature columns na best pipeline: ", self.fitted_feature_columns)
        print("Het type van best model pipeline na training van model:", type(self.best_linreg_pipeline))
        print("best model komt hier aan: ", self.best_linreg_pipeline)

        # Save the best pipeline with joblib
        dump(self.best_linreg_pipeline, 'best_linreg_pipeline.joblib')     
        dump(self.fitted_feature_columns, 'fitted_feature_columns.joblib')

        return

    def map_province_to_onehot(self):
        """
        self.province_columns=[province_Brabant_Wallon, province_Brussels, province_East_Flanders, province_Flemish_Brabant, province_Hainaut, province_Liege, province_Limburg, province_Luxembourg, province_Namur, province_West_Flanders]
       
        province_to_onehot = {'province_Brabant_Wallon': [1,0,0,0,0,0,0,0,0,0],
        'province_Brussels': [0,1,0,0,0,0,0,0,0,0], 
        province_East_Flanders: [0,0,1,0,0,0,0,0,0,0], 
        province_Flemish_Brabant: [0,0,0,1,0,0,0,0,0,0], 
        province_Hainaut: [0,0,0,0,1,0,0,0,0,0], 
        province_Liege: [0,0,0,0,0,1,0,0,0,0], 
        province_Limburg: [0,0,0,0,0,0,1,0,0,0], 
        province_Luxembourg: [0,0,0,0,0,0,0,1,0,0], 
        province_Namur: [0,0,0,0,0,0,0,0,1,0], 
        province_West_Flanders]: [0,0,0,0,0,0,0,0,0,1]}
        """
        for key, value in self.province_to_onehot.items():
            if self.df.loc[0,'province'] == key:
                province_df = pd.DataFrame(np.array(value).reshape(1,-1), columns=self.province_columns)
        self.df = pd.concat([self.df, province_df], axis = 1)
              
        return self.df

    def predict_new_price(self):
        
        print("df at start of predict_new_price: ", self.df)
        

        # Convert the province to a OneHot code
        self.df = self.map_province_to_onehot()
        print("df na map province to onehot en terug in predict_new_price method: ", self.df)

        # Load feature columns and model pipeline:
        self.fitted_feature_columns = load('fitted_feature_columns.joblib')     
        self.loaded_pipeline = load('best_linreg_pipeline.joblib')


        print("feature columns after loading from joblib in predict_new_price:", self.fitted_feature_columns)

        self.X = self.df[self.fitted_feature_columns]                   # fit the input df to the df used for model fitting
        print("self X na aanpassen aan fitted_feature_columns", self.X)
        
        self.prediction = self.loaded_pipeline.predict(self.X)
        print("Type of predictions: ", type(self.prediction))
        
        # Save the predictions as csv file to have quick view
        np.savetxt("predictions.csv", self.prediction, delimiter=",", fmt="%.2f")

        return self.prediction

    def evaluation(self, y_test, y_pred):

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

    def train_workflow(self): # dit kan ook pipeline genoemd worden
        self.X, self.y = self.create_Xy()
        self.ordinal_encoding()
        self.onehot_encoding()
        print("Self.X na onehot endoing:", type(self.X), self.X.head())

        self.best_linreg_model_pipeline(self.X, self.y)

        return