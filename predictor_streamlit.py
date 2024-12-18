import streamlit as st
import time
import requests

# Sidebar
st.sidebar.title("Immo Eliza")
st.sidebar.image(r"BeCodeHarmonyRealEstateLogo.jpg", width=100)

st.sidebar.subheader('Web Scraper')
st.sidebar.subheader('Data Cleaner')
st.sidebar.subheader('Data Analyzer')
st.sidebar.subheader('Modeller')
st.sidebar.subheader('Predictor')

# Input parameters
container1 = st.container()
col1,col2, col3=container1.columns([1,1,1])
#col2.image(r"assets\house.jpg", width = 200)
col2.subheader("Predictor")
#container1.write("Welcome to the prediction tool of Immo Eliza. Below you can enter the information of the property you want to buy or sell. The prediction tool will return an estimate of what the ask price on ImmoWeb can be.")

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

house_options= ['House', 'Appartment']
radio_type = col1.radio('Type', house_options, key = 'radio_type', index = 0)

facade_options = ['Rijhuis', 'Half open bouw', 'Open bouw']
facade_count_selected = col3.radio('Facades', facade_options, key = 'radio_nr_of_facades')

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

col1.text_input('Postal code', max_chars = 4, placeholder = "bv. 9090", key = 'input_postal_code')

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

net_habitable_surface = col1.slider('Net habitable surface (m²)',  0, 2000, key = 'slider_net_habitable_surface')
land_surface = col3.slider('Plot size (m²)',  0, 2000, key = 'slider_land_surface')

epc_ord_enc_options = ['F', 'E', 'D', 'C', 'B', 'A']
epc_ord_enc_selected = col3.select_slider('EPC', epc_ord_enc_options, key = 'slider_epc')

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

building_condition_ord_enc_options = ['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']
building_condition_ord_enc_selected = col1.radio("Condition of the building", building_condition_ord_enc_options, key='radio_building_condition')

kitchen_type_ord_enc_options = ['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']
kitchen_type_ord_enc_selected = col3.radio("State of the kitchen", kitchen_type_ord_enc_options, key='radio_kitchen_type')

container = st.container()
col1, col2, col3 = container.columns([1, 1, 1])  # Adjust proportions of columns

# Trigger the POST request when the button is clicked
if col2.button('Submit for prediction', icon=":material/query_stats:"):
    # Define the payload
    #data = [bedroom_count, net_habitable_surface, facade_count, land_surface, has_assigned_city_10, kitchen_type_ord_enc, building_condition_ord_enc, epc_ord_enc]
    
    #payload = {"bedroom_count": bedroom_count, 
    #           "net_habitable_surface": net_habitable_surface, 
    #           "facade_count": facade_count, 
    #           "land_surface": land_surface, 
    #           "has_assigned_city_10": has_assigned_city_10, 
    #           "kitchen_type_ord_enc": kitchen_type_ord_enc, 
    #           "building_condition_ord_enc": building_condition_ord_enc, 
    #           "epc_ord": epc_ord_enc
    #           }

#with st.spinner('Wait for it...'):
#    time.sleep(10)  # Simulating a process delay
#container.success("You did it!")

# Display maps
import pandas as pd
import numpy as np
import streamlit as st

df = pd.DataFrame(
    np.random.randn(500, 2) / [50, 50] + [51.05, 3.71667], columns=['lat', 'lon']
)
st.map(df)
