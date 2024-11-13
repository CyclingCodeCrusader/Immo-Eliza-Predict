import streamlit as st
import requests


# Sidebar
st.sidebar.title("Immo Eliza")
st.sidebar.image(r"BeCodeHarmonyRealEstateLogo.jpg", width=100)

st.sidebar.subheader('Web Scraper')
st.sidebar.subheader('Data Cleaner')
st.sidebar.subheader('Data Analyzer')
st.sidebar.subheader('Modeller')
st.sidebar.subheader('Predictor')


# Define options for the radio button
options1 = ["Option 1", "Option 2", "Option 3"]
options2 = ["Option 1", "Option 2", "Option 3"]

# Input parameters
container1 = st.container()
col1,col2, col3=container1.columns([1,1,1])
#col2.image(r"assets\house.jpg", width = 200)
col2.subheader("Predictor")
#container1.write("Welcome to the prediction tool of Immo Eliza. Below you can enter the information of the property you want to buy or sell. The prediction tool will return an estimate of what the ask price on ImmoWeb can be.")

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

# Selection of type of property. NOte that nothing is done with this, as there are only houses in the database I use.
house_options = ['House', 'Appartment']
radio_type = col1.radio('Type', house_options, key = 'radio_type', index = 0)

facade_options = ['Rijhuis', 'Half open bouw', 'Open bouw']
facade_count_selected = col3.radio('Facades', facade_options, key = 'radio_nr_of_facades')
facade_count = facade_options.index(facade_count_selected)+2 # add 2 to go from index to the number of facades

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

locality_code = col1.number_input('Postal code', value = 8000, key = 'input_postal_code1')

has_assigned_city_10 = col3.checkbox("Close to a major city", 'Yes', key = "checkbox_city")

#list_of_cities = ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liège', 'Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas']
#col1.write("List of the cities:")
#col1.write(str(list_of_cities))
#ordinals_kitchen = [['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']]  # Order for each ordinal column
#ordinals_building_condition = [['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']]  # Order for each ordinal column
#ordinals_epc = [['F', 'E', 'D', 'C', 'B', 'A']]  # Order for each ordinal column

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

bedroom_count_options = [1, 2, 3, 4, 5]
bedroom_count_selected = col1.select_slider('Number of bedrooms', bedroom_count_options, key = 'slider_nr_of_bedrooms')
bedroom_count = bedroom_count_options.index(bedroom_count_selected)+1 # add 1 to go from index to the number of facades

net_habitable_surface = col1.slider('Net habitable surface (m²)',  0, 2000, key = 'slider_net_habitable_surface')

land_surface = col3.slider('Plot size (m²)',  0, 2000, key = 'slider_land_surface')

epc_ord_enc_options = ['F', 'E', 'D', 'C', 'B', 'A']
epc_ord_enc_selected = col3.select_slider('EPC', epc_ord_enc_options, key = 'slider_epc')
epc_ord_enc = epc_ord_enc_options.index(epc_ord_enc_selected)+1 # add 1 to go from index to the number of facades

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

building_condition_ord_enc_options = ['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
building_condition_ord_enc_selected = col1.radio("Condition of the building", building_condition_ord_enc_options, key='radio_building_condition')
building_condition_ord_enc = building_condition_ord_enc_options.index(building_condition_ord_enc_selected)+1 # add 1 to go from index to the number of facades

kitchen_type_ord_enc_options = ['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']
kitchen_type_ord_enc_selected = col3.radio("State of the kitchen", kitchen_type_ord_enc_options, key='radio_kitchen_type')
kitchen_type_ord_enc = kitchen_type_ord_enc_options.index(kitchen_type_ord_enc_selected)+1 # add 1 to go from index to the number of facades

container = st.container()
col1, col2, col3 = container.columns([1, 1, 1])  # Adjust proportions of columns

# Define the FastAPI backend URL
API_URL = "http://localhost:8000/api/predict"

# Display a button to submit the selection
if col2.button('Submit for prediction', icon=":material/query_stats:"):
    # Prepare the payload with the index
    payload = {"bedroom_count": bedroom_count, 
               "net_habitable_surface": net_habitable_surface, 
               "facade_count": facade_count, 
               "land_surface": land_surface, 
               #"has_assigned_city_10": has_assigned_city_10,
               "epc_ord_enc": epc_ord_enc, 
               "building_condition_ord_enc": building_condition_ord_enc, 
               "kitchen_type_ord_enc": kitchen_type_ord_enc, 
               "locality_code": locality_code}
    
    # Send a POST request to the FastAPI backend
    response = requests.post(API_URL, json=payload)
    
    # Display the backend's response
    if response.status_code == 200:
        st.write("Response from API:", response.json()["message"])
    else:
        st.write("Error:", response.status_code)
