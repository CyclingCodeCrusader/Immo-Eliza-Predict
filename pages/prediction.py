# This file contains the streamlit code for the quick prediction tool.

import streamlit as st
import requests

# Page header
container = st.container()
col1, col2 = container.columns([1, 5])  # Adjust proportions of columns
col1.image(r"assets\BeCodeHarmonyRealEstateLogo.jpg", width=60)
col2.subheader("House price estimation", divider="rainbow")

# Input parameters
container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

# Selection of type of property. Note that nothing is done with this, as there are only houses in the database I use.
house_options = ['House', 'Appartment']
radio_type = col1.radio('Type', house_options, key = 'radio_type_2', index = 0)

facade_options = ['Rijhuis', 'Half open bouw', 'Open bouw']
facade_count_selected = col3.radio('Facades', facade_options, key = 'radio_nr_of_facades')
facade_count = facade_options.index(facade_count_selected)+2 # add 2 to go from index to the number of facades

locality_code = col4.number_input('Postal code', value = 8000, key = 'input_postal_code1')

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

bedroom_count_options = [1, 2, 3, 4, 5]
bedroom_count_selected = col1.select_slider('Number of bedrooms', bedroom_count_options, key = 'slider_nr_of_bedrooms')
bedroom_count = bedroom_count_options.index(bedroom_count_selected)+1 # add 1 to go from index to the number of facades

net_habitable_surface = col3.number_input('Net habitable surface (m²)', value = 200, key = 'input_net_habitable_surface')

land_surface = col3.number_input('Plot size (house, garden and terrace)(m²)', value = 100, key = 'input_land_surface')

epc_ord_enc_options = ['F', 'E', 'D', 'C', 'B', 'A']
epc_ord_enc_selected = col1.select_slider('EPC', epc_ord_enc_options, key = 'slider_epc')
epc_ord_enc = epc_ord_enc_options.index(epc_ord_enc_selected)+1 # add 1 to go from index to the number of facades

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

building_condition_ord_enc_options = ['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
building_condition_ord_enc_selected = col1.radio("Condition of the building", building_condition_ord_enc_options, key='radio_building_condition')
building_condition_ord_enc = building_condition_ord_enc_options.index(building_condition_ord_enc_selected)+1 # add 1 to go from index to the number of facades

kitchen_type_ord_enc_options = ['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']
kitchen_type_ord_enc_selected = col3.radio("State of the kitchen", kitchen_type_ord_enc_options, key='radio_kitchen_type')
kitchen_type_ord_enc = kitchen_type_ord_enc_options.index(kitchen_type_ord_enc_selected)+1 # add 1 to go from index to the number of facades


st.divider()
container = st.container()
col1, col2 = container.columns([1, 1])  # Adjust proportions of columns

# Selection of model.
model_options = ['Use all trained models', 'Best linear regression model', 'Best polynomial regression model', 'Best tree-based regression model']
selected_model = col1.radio('Select prediction model', model_options, key = 'model_type', index = 0)

# Define the FastAPI backend URL
API_URL = "http://localhost:8000/api/predict"

# Display a button to submit the selection
if col1.button('Submit for prediction', icon=":material/query_stats:"):
    # Prepare the payload with the index
    payload = {"bedroom_count": bedroom_count, 
               "net_habitable_surface": net_habitable_surface, 
               "facade_count": facade_count, 
               "land_surface": land_surface,
               "epc_ord_enc": epc_ord_enc, 
               "building_condition_ord_enc": building_condition_ord_enc, 
               "kitchen_type_ord_enc": kitchen_type_ord_enc, 
               "locality_code": locality_code,
               "selected_model": selected_model}
    
    # Send a POST request to the FastAPI backend
    response = requests.post(API_URL, json=payload)

    # Display the backend's response
    if response.status_code == 200:
        col2.write("Price estimation:")
        unpack = response.json()["message"]
        for key, value in unpack.items():
            col2.write(f"{key}: € {value:,.0f}".replace(",", " "))
            
        #st.subheader(response.json()["message"])        
        
    else:
        st.write("Error:", response.status_code)
