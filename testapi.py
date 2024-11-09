from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

from functions.utils import open_pkl_as_dataframe
from functions.modeller_functions import ordinal_encoding, OneHot_encoding
from functions.modeller_functions import models_linear, models_polynomial, models_treebased, create_Xy, polynomial_simple, XGBoost
from functions.modeller_functions import save_best_model, load_prediction_model


# Creating FastAPI instance
app = FastAPI()
 
@app.get("/")
async def root():
    """Route that returns 'Alive!' if the server runs."""
    return {"Status": "Alive! Welcome ! This is local from VSCODE and FastApi."}

@app.post('/hello/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to testapi via POST!, {name}'}

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

# Define a Pydantic model for the expected JSON structure
class SalaryData(BaseModel):
    salary: int
    bonus: int
    taxes: int

# Endpoint using Pydantic model
@app.post("/salary")
async def results(data: SalaryData):
    # The data is automatically validated and parsed into `SalaryData` instance
    result = data.salary + data.bonus - data.taxes
    return {"message": "Data received successfully", "input": data, "result": result}

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
 
# Loading Iris Dataset
iris = load_iris()
 
# Getting our Features and Targets
X = iris.data
Y = iris.target
 
# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X,Y)

# HIER KAN DUS HET MODEL INGELADEN WORDEN VAN OP EEN JOBLIB FILE EN IN EEN VARIABELE GESTOEKEN WORDEN
# BV VARIABELE MODEL =

# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict2')
def predict2(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
     
    # Predicting the Class
    class_idx = clf.predict(test_data)[0]
     
    # Return the Result
    return { 'class' : iris.target_names[class_idx]}