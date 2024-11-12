# streamlit_app.py
import streamlit as st
import requests



container = st.container()
col1, col2, col3, col4 = container.columns([1, 1, 1, 1])  # Adjust proportions of columns
radio_type = col1.radio('Type', ['House', 'Appartment'], key = 'radio_type', index = 0)

# Display the selected option
st.write("You selected:", radio_type)

# Define the backend API URL
API_URL = "http://localhost:8000/api/post_data"

# User input
data1 = st.text_input("Enter some data to send to the backend", key = "data1")
data2 = st.text_input("Enter some data to send to the backend", key = "data2")

# Trigger the POST request when the button is clicked
if st.button("Send to Backend"):
    # Define the payload
    payload = {"data1": data1, "data2": data2}
    
    # Send a POST request to the backend API
    response = requests.post(API_URL, json=payload)
    
    # Handle the response
    if response.status_code == 200:
        st.write("Response from backend:", response.json())
    else:
        st.write("Error:", response.status_code)