
import pandas as pd
import joblib
import pickle
import numpy as np


from functions.utils import open_pkl_as_dataframe
from functions.modeller_functions import ordinal_encoding, OneHot_encoding
from functions.modeller_functions import models_linear, models_polynomial, models_treebased, create_Xy, polynomial_simple, XGBoost
from functions.modeller_functions import save_best_model, load_prediction_model


# Load the model from the file
loaded_pipeline = joblib.load('best_model_pipeline.joblib')

# Create a new house 
feature_names = ['bedroom_count', 'net_habitable_surface', 'facade_count', 'land_surface', 'has_assigned_city_10', 
                 'kitchen_type_ord_enc', 'building_condition_ord_enc', 'epc_ord_enc', 
                 'province_Brabant_Wallon', 'province_Brussels', 'province_East_Flanders', 'province_Flemish_Brabant', 'province_Hainaut', 
                 'province_Liege', 'province_Limburg', 'province_Luxembourg', 'province_Namur', 'province_West_Flanders']

new_house = np.array([3,200,4,500,1,3,3,3,0,0,1,0,0,0,0,0,0,0]).reshape(1,-1)

df_new_house= pd.DataFrame(new_house, columns=feature_names)
print(df_new_house)

# Use the loaded model to make predictions
predictions = loaded_pipeline.predict(df_new_house)

# Save the predictions as csv file to have quick view
np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%.2f")