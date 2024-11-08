import pickle
import pandas as pd
import joblib

from functions.utils import open_pkl_as_dataframe
from functions.modeller_functions import ordinal_encoding, OneHot_encoding
from functions.modeller_functions import models_linear, models_polynomial, models_treebased, create_Xy, polynomial_simple, XGBoost
from functions.modeller_functions import save_best_model, load_prediction_model

df1 = open_pkl_as_dataframe(default_path = r'data\after_analysis.pkl')# Fill your path to file

# Part of modeling starts here
# Overview and grouping of the datacolumns for loop later on
numerical_columns = ['bedroom_count', 'net_habitable_surface', 'facade_count','land_surface']

numerical_columns_backlog = ['terrace_surface','garden_surface'] # ,'landSurface'
unnecessary_columns = ['id','locality_name','street', 'number','locality_latitude','locality_longitude']
derivative_columns = ['price_per_sqm','price_per_sqm_land']
categorical_columns = ['subtype','kitchen_type','building_condition','epc','locality_code','province', 'assigned_city','assigned_city_5', 'assigned_city_10', 'assigned_city_15']
label_encoded_columns = ['kitchen_type_encoded', 'building_condition_encoded','epc_encoded']
binary_columns = ['pool', 'fireplace','furnished', 'has_assigned_city','has_assigned_city_5', 'has_assigned_city_10', 'has_assigned_city_15'] # 'hasTerrace', not reliably maintained so leaving it out of analyzing/visualization

to_encode_columns = ['kitchen_type','building_condition','epc']

# Save our target column in the variable y, and the predictor columns in the variable X
target_column = ['price']
predictor_columns = numerical_columns + to_encode_columns + ['province'] + ['has_assigned_city_10']
# Create feature matrix X and target matrix y. This is done here, 
# because it is easier to perform label encoding and onehot encoding on the feature matrix instaed of the full dataframe, 
# so that you don't have to include all the newly generated encoded columns 
X, y = create_Xy(df1, predictor_columns, target_column)

# Part on label encoding
X = ordinal_encoding(X, to_encode_columns)

# Part on one hot encoding
X = OneHot_encoding(X, columns = ['province'])

print(X.info())

best_pipeline = models_linear(X,y)
#best_model, best_params = models_polynomial(X,y)
#polynomial_simple(X,y)
#models_treebased(X,y)
#XGBoost(X,y)

# Save the best pipeline with joblib
joblib.dump(best_pipeline, 'best_model_pipeline.joblib')