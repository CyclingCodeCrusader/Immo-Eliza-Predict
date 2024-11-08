import streamlit as st
import time

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

col1.radio('Type', ['House', 'Appartment'], key = 'radio_type')
col3.radio('Facades', ['Rijhuis', 'Half open bouw', 'Open bouw'], key = 'radio_nr_of_facades')

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns
col1.text_input('Postal code', max_chars = 4, placeholder = "bv. 9090", key = 'input_postal_code')
col3.checkbox("Close to a major city", 'Yes', key = "checkbox_city")


#list_of_cities = ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liège', 'Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas']
#col1.write("List of the cities:")
#col1.write(str(list_of_cities))
#ordinals_kitchen = [['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped']]  # Order for each ordinal column
#ordinals_building_condition = [['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']]  # Order for each ordinal column
#ordinals_epc = [['F', 'E', 'D', 'C', 'B', 'A']]  # Order for each ordinal column

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns

col1.select_slider('Number of bedrooms', [1, 2, 3, 4, 5], key = 'slider_nr_of_bedrooms')
col1.slider('Net habitable surface (m²)',  0, 2000, key = 'slider_net_habitable_surface')
col3.slider('Plot size (m²)',  0, 2000, key = 'slider_land_surface')
col3.select_slider('EPC', ['F', 'E', 'D', 'C', 'B', 'A'], key = 'slider_epc')

container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns
col1.radio("Condition of the building", ['To renovate', 'To be done up', 'Good', 'Just renovated', 'As new'], key='radio_building_condition')
col3.radio("State of the kitchen", ['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped'], key='radio_kitchen_type')

container = st.container()
col1, col2, col3 = container.columns([1, 1, 1])  # Adjust proportions of columns
col2.button('Submit for prediction', icon=":material/query_stats:")

#with st.spinner('Wait for it...'):
#    time.sleep(10)  # Simulating a process delay
#container.success("You did it!")
#container.balloons()  # Celebration balloons

# Display maps
import pandas as pd
import numpy as np
import streamlit as st

df = pd.DataFrame(
    np.random.randn(500, 2) / [50, 50] + [51.05, 3.71667], columns=['lat', 'lon']
)
st.map(df)
