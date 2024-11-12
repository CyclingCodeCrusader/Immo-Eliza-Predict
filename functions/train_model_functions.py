from sklearn.preprocessing import OrdinalEncoder

import seaborn as sns
import geopandas as gpd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def create_Xy(df, predictor_columns, target_column):

    X = df[predictor_columns]
    y = df[target_column]

    return X, y

def ordinal_encoding(X, to_encode_columns):
    #This categorical data has a natural order we encode it in a way that reflects this ordering. We will use ordinal Encoding.
    
    # Define the custom order for the 'Kitchen_type' column
    ordinals_kitchen = [['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']]  # Order for each ordinal column
    ordinals_building_condition = [['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']]  # Order for each ordinal column
    ordinals_epc = [['F', 'E', 'D', 'C', 'B', 'A']]  # Order for each ordinal column

    #to_encode_columns = ['kitchen_type','state_of_building','epc']
    ordinals_list = [ordinals_kitchen, ordinals_building_condition, ordinals_epc]
    ordinal_encoded_columns = []
    #print(type(X))

    for i, col in enumerate(to_encode_columns):
        # Initialize OrdinalEncoder with the specified categories
        encoder = OrdinalEncoder(categories=ordinals_list[i])
        name_ord_enc = f"{col}_ord_enc"
        ordinal_encoded_columns.append(name_ord_enc)
        # Fit and transform the column
        #print(col, type(X[[col]]))
        #X[name_ord_enc] = encoder.fit_transform(X[[col]]) # syntax from solution of error message dfmi.loc[:, ('one', 'second')]

        X = X.assign(**{name_ord_enc: encoder.fit_transform(X[[col]])})

        #f"{col}_ord_enc" name_ord_enc
        X = X.drop(columns = col)

    return X

def OneHot_encoding(X, columns):
    
    for col in columns:
        # One-hot encode in the dataframe
        X = pd.get_dummies(X, columns=[col], drop_first=True)

    return X

def evaluation(y_test, y_pred):

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

def save_best_model(best_model, file_path):
    import joblib
    joblib.dump(best_model, file_path)
    print('Best model: ', best_model, "saved as: ", file_path)
    return

def load_prediction_model(X_test, model_file_path, params_file_path):
    import joblib

    # Load the model from the file
    loaded_model = joblib.load(file_path)

    # Use the loaded model to make predictions
    predictions = loaded_model.predict(X_test)
    print("Predictions:", predictions)

    return predictions

def models_linear(X,y):        
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
    evaluation(y_test.to_numpy(), y_pred)
    print(X_train)

    return best_pipeline

def models_polynomial(X,y):
        
    # Create the pipeline with a placeholder for the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('polynomial_features', PolynomialFeatures()), 
        ('regressor', None)  # Placeholder for the model
    ])

    # Define the models and parameters to explore
    param_grid = [
        {'polynomial_features__degree': [1, 2 ,3]},   # Choose degree range based on data
        #{'regressor': [LinearRegression()]},                    # No hyperparameters for LinearRegression
        #{'regressor': [Ridge()],
        #'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000],
        #'regressor__solver': ["auto","cholesky","sparse_cg"]}, #"lbfgs",
        #{'regressor': [Lasso()],
        #'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000]},
        #{'regressor': [ElasticNet()],
        #'regressor__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
        ]

    # Define KFold cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up GridSearchCV with the pipeline, parameter grid, and KFold cross-validation
    grid_search = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Display the best model and parameters
    print("PERFORMANCE OF POLYNOMIAL MODELS: \n ----LinearRegression with PolynomialFeatures \n ----regularization: Ridge, Lasso, ElasticNet")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("Best model:", best_model)
    print("Best parameters:", best_params)

    print("Best model score on training set: ", best_model.score(X_train, y_train))
    print("Best model score on test set: ", best_model.score(X_test, y_test))

    # Predict on test data using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    evaluation(y_test.to_numpy(), y_pred)

    return best_model, best_params

def polynomial_simple(X,y):
    degree = 2
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Define KFold cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    polyreg.fit(X_train, y_train)

    best_model = polyreg
    best_params = {}

    print("Score on training set: ", polyreg.score(X_train, y_train))
    print("Score on test set: ", polyreg.score(X_test, y_test))

    y_pred = polyreg.predict(X_test)

    """plt.figure()
    plt.scatter(X_train['net_habitable_surface'], y_train, color = "green")
    plt.scatter(X_test['net_habitable_surface'], y_test, color = "blue")
    plt.scatter(X_test['net_habitable_surface'], y_pred, color="red")
    plt.title("Polynomial regression with degree "+ str(degree))
    plt.show()"""

    # Evaluate the model
    evaluation(y_test.to_numpy(), y_pred)

    return best_model, best_params

def models_treebased(X,y):
        
    # Create the pipeline with a placeholder for the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', None)  # Placeholder for the model
    ])

    # Define the models and parameters to explore
    param_grid = [
        #{'regressor': [RandomForestRegressor()]},
        #{'regressor': [DecisionTreeRegressor()]},
        #{'regressor': [CatBoostRegressor()], 'regressor__iterations': [1, 5, 25, 125], 'regressor__depth': [3], 'regressor__learning_rate': [0.01, 0.1, 1], 'regressor__loss_function':['RMSE']},
        {'regressor': [XGBRegressor()], 'regressor__max_depth': [3], 'regressor__eta': [0.01, 0.1, 1], 'regressor__objective': ['binary:logistic'], 'regressor__tree_method': ['hist'], 'regressor__device':['cuda']},
        #{'regressor': [SVR()], 'regressor__kernel':['linear', 'poly','rbf','sigmoid','precomputed'], 'regressor__degree': [2, 3], 'regressor__gamma':['scale','auto','float']}
        ]

    # Define KFold cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up GridSearchCV with the pipeline, parameter grid, and KFold cross-validation
    grid_search = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Display the best model and parameters
    print("PERFORMANCE OF TREE-BASED MODELS: \n ----RandomForestRegressor, DecisionTreeRegressor, CatBoostRegressor, XGBRegressor, SVR")
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("Best model:", best_model)
    print("Best parameters:", best_params)

    print("Best model score on training set: ", best_model.score(X_train, y_train))
    print("Best model score on test set: ", best_model.score(X_test, y_test))

    # Predict on test data using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    evaluation(y_test.to_numpy(), y_pred)
    
    return best_model, best_params

def XGBoost(X,y):

            
    # Create the pipeline with a placeholder for the model
    """pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', [XGBRegressor()])  # Placeholder for the model
    ])"""

    param_grid = {
    'colsample_bytree': [0, 0.5,1.0],
    'gamma': [0, 10, 100, 1000],
    'learning_rate': [0, 0.5, 1.0],
    'max_depth': [5],
    'n_estimators': [300],
    'reg_alpha': [0, 10, 10, 1000],
    'reg_lambda': [0, 10, 10, 1000],
    'subsample': [0.25, 0.5, 0.75, 1],
    #'num_boosted_rounds': [50],
    #'early_stopping_rounds': [20]
    }

    # Initialize the XGBoost model with best parameters and GPU configuration
    best_xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist',  # Enables GPU usage
    device="cuda"
    )
    # Define KFold cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up GridSearchCV with the pipeline, parameter grid, and KFold cross-validation
    grid_search = GridSearchCV(estimator = best_xgb_model, param_grid = param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Display the best model and parameters
    print("PERFORMANCE OF TREE-BASED MODELS: \n ----RandomForestRegressor, DecisionTreeRegressor, CatBoostRegressor, XGBRegressor, SVR")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("Best model:", best_model)
    print("Best parameters:", best_params)

    print("Best model score on training set: ", best_model.score(X_train, y_train))
    print("Best model score on test set: ", best_model.score(X_test, y_test))

    # Predict on test data using the best model
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    evaluation(y_test.to_numpy(), y_pred)
    
    return best_model, best_params

