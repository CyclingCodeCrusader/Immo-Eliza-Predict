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
    #has_assigned_city_10 : bool
    kitchen_type_ord_enc : int
    building_condition_ord_enc : int
    epc_ord_enc : int
    locality_code: int

# Endpoint to receive the radio button selection index
@app.post("/api/predict")
def receive_selection(selection: SelectionModel):
    # Process the index (e.g., create a response message)
    response_message1 = f"You selected option number: {selection.bedroom_count} and {selection.facade_count} and {selection.net_habitable_surface} and {selection.land_surface}. Ordinal selected are: {selection.kitchen_type_ord_enc} and {selection.building_condition_ord_enc} and {selection.epc_ord_enc}"
    
    # Making the data in a form suitable for prediction
    test_data = [[
            selection.bedroom_count, 
            selection.net_habitable_surface,
            selection.facade_count, 
            selection.land_surface, 
            #selection.has_assigned_city_10, 
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

class province_input(BaseModel):
    code : int
    #province: int

@app.post('/api/province')
def province(input: province_input):
    code = input.code
    if code == 9000:
        name = "Gantoise"
    return {"message": "Data received successfully", "input": code, "result": name} #


class predict_request_body(BaseModel):
    bedroom_count : int
    net_habitable_surface: float
    facade_count : int
    land_surface: float
    has_assigned_city_10 : bool
    kitchen_type_ord_enc : int
    building_condition_ord_enc : int
    epc_ord_enc : int


@app.post('/predict')
def predict(data : predict_request_body):

    # Making the data in a form suitable for prediction
    test_data = [[
            data.bedroom_count, 
            data.net_habitable_surface, 
            data.facade_count,
            data.land_surface, 
            data.has_assigned_city_10, 
            data.kitchen_type_ord_enc, 
            data.building_condition_ord_enc, 
            data.epc_ord_enc

    ]]
    # Extract the data from the request body
    received_data = data.bedroom_count
    print(f"Received data: {received_data}")  # For logging purposes

    # Perform any processing here
    #processed_data = received_data.upper()  # Example processing

    # Return a response
    return {"received": received_data}
    
    # Load the model from the file
    #loaded_pipeline = joblib.load('best_model_pipeline.joblib')

    # Create a new house 
    #feature_names = ['bedroom_count', 'net_habitable_surface', 'facade_count', 'land_surface', 'has_assigned_city_10', 
    #                'kitchen_type_ord_enc', 'building_condition_ord_enc', 'epc_ord_enc']
    #, 
    #                'province_Brabant_Wallon', 'province_Brussels', 'province_East_Flanders', 'province_Flemish_Brabant', 'province_Hainaut', 
    #                'province_Liege', 'province_Limburg', 'province_Luxembourg', 'province_Namur', 'province_West_Flanders']

    #new_house = np.array([3,200,4,500,1,3,3,3,0,0,1,0,0,0,0,0,0,0]).reshape(1,-1)
    #new_house = np.array(test_data).reshape(1,-1)

    #df_new_house= pd.DataFrame(new_house, columns=feature_names)
    #print(df_new_house)

    # Use the loaded model to make predictions
    #predictions = loaded_pipeline.predict(df_new_house)
    #print("Type of predictions: ", type(predictions))
    # Save the predictions as csv file to have quick view
    #np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%.2f")

    # Return the Result
    #return { 'Price prediction in euro:' : round(predictions[0])}
    return { "message": "Data received successfully", "input": test_data, "result": result}