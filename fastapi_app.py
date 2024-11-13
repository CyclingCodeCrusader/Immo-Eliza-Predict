# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from functions.preprocessing_functions import map_province, get_province
from scripts.processor import Processor
from scripts.modeller import Modeller

app = FastAPI()

# Define the request model
# Creating class to define the request body
# and the type hints of each attribute
class SelectionModel(BaseModel):
    bedroom_count : int
    net_habitable_surface: float
    facade_count : int
    land_surface: float
    kitchen_type_ord_enc : int
    building_condition_ord_enc : int
    epc_ord_enc : int
    locality_code: int

# Endpoint to receive the radio button selection index
@app.post("/api/predict")
def receive_selection(selection: SelectionModel):
    # Process the index (e.g., create a response message)
    response_message1 = f"You have selected the following options:\n -Bedrooms: {selection.bedroom_count}\n -Facades: {selection.facade_count}\n -Net habitable surface: {selection.net_habitable_surface}\n -Plot surface: {selection.land_surface}\n -Kitchen type: {selection.kitchen_type_ord_enc}\n -Building condition: {selection.building_condition_ord_enc}\n -EPC: {selection.epc_ord_enc}"
    
    # Making the data in a form suitable for prediction
    test_data = [[
            selection.bedroom_count, 
            selection.net_habitable_surface,
            selection.facade_count, 
            selection.land_surface,
            selection.kitchen_type_ord_enc, 
            selection.building_condition_ord_enc, 
            selection.epc_ord_enc,
            selection.locality_code
            ]]
    print(type(test_data), test_data)

    # Call methods from the Processor class and return a df
    predict_input_processor = Processor()
    df = predict_input_processor.predict_workflow(test_data)
    print(df)

    # Call methods from the Modeller class and return a prediction to pass to the message output for api/streamlit
    predictor = Modeller(df)
    prediction = predictor.predict_new_price()
    
    response_message = f"{response_message1} Price prediction in euro: {round(prediction[0])}"    # Generate a response message with info on the input, and the price prediction
    
    return {"message": response_message}                                                          # Return the Result

@app.post('/hello/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to testapi via POST!, {name}'}

@app.post('/api/province')
def province(input: province_input):
    code = input.code
    if code == 9000:
        name = "Gantoise"
    return {"message": "Data received successfully", "input": code, "result": name} #
