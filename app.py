from fastapi import FastAPI
import os
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

from test import value100

# Set port to the env variable PORT to make it easy to choose the port on the server
# If the Port env variable is not set, use port 8000
PORT = os.environ.get("PORT", 8000)
app = FastAPI(port=PORT)


@app.get("/")
async def root():
    """Route that returns 'Alive!' if the server runs."""
    return {"Status": "Alive! Welcome !"}

@app.get("/multi/{value}")
async def multiply(value: int):
    value = value100(value)
    return {"result": value - 1}
# Defining path operation for /name endpoint

@app.get('/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to GeeksforGeeks!, {name}'}

# Creating class to define the request body
# and the type hints of each attribute
class predict_request_body(BaseModel):
    bedroom_count : float
    net_habitable_surface : float
    facade_count : float
    land_surface : float
    has_assigned_city_10 : bool
    kitchen_type_ord_enc : float
    building_condition_ord_enc : float
    epc_ord_enc : float
    province_Brabant_Wallon : bool
    province_Brussels : bool
    province_East_Flanders : bool
    province_Flemish_Brabant : bool
    province_Hainaut : bool
    province_Liege : bool
    province_Limburg : bool
    province_Luxembourg : bool
    province_Namur : bool
    province_West_Flanders : bool
    

@app.post('/predict')
async def predict(data : predict_request_body):

    # Making the data in a form suitable for prediction
    test_data = [[
            data.bedroom_count, 
            data.net_habitable_surface, 
            data.facade_count, data.land_surface, 
            data.has_assigned_city_10, 
            data.kitchen_type_ord_enc, 
            data.building_condition_ord_enc, 
            data.epc_ord_enc, 
            data.province_Brabant_Wallon, 
            data.province_Brussels, 
            data.province_East_Flanders, 
            data.province_Flemish_Brabant, 
            data.province_Hainaut, 
            data.province_Liege, 
            data.province_Limburg, 
            data.province_Luxembourg, 
            data.province_Namur, 
            data.province_West_Flanders
    ]]

    # Load the model from the file
    loaded_pipeline = joblib.load('best_model_pipeline.joblib')

    # Create a new house 
    feature_names = ['bedroom_count', 'net_habitable_surface', 'facade_count', 'land_surface', 'has_assigned_city_10', 
                    'kitchen_type_ord_enc', 'building_condition_ord_enc', 'epc_ord_enc', 
                    'province_Brabant_Wallon', 'province_Brussels', 'province_East_Flanders', 'province_Flemish_Brabant', 'province_Hainaut', 
                    'province_Liege', 'province_Limburg', 'province_Luxembourg', 'province_Namur', 'province_West_Flanders']

    #new_house = np.array([3,200,4,500,1,3,3,3,0,0,1,0,0,0,0,0,0,0]).reshape(1,-1)
    new_house = np.array(test_data).reshape(1,-1)

    df_new_house= pd.DataFrame(new_house, columns=feature_names)
    print(df_new_house)

    # Use the loaded model to make predictions
    predictions = loaded_pipeline.predict(df_new_house)
    print("Type of predictions: ", type(predictions))
    # Save the predictions as csv file to have quick view
    np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%.2f")

    # Return the Result
    return { 'Price prediction in euro:' : round(predictions[0])}