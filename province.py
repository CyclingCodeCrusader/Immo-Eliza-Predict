import streamlit as st
import requests


# Define the FastAPI backend URL
API_URL = "http://localhost:8000/api/province"
code = st.number_input('Postal code', value = 8000, key = 'input_postal_code1')  

# Trigger the POST request when the button is clicked
if st.button("Send to Backend"):
    # Define the payload
    payload = {"code": code}
    
    # Send a POST request to the backend API
    response = requests.post(API_URL, json=payload)
    
    # Handle the response
    if response.status_code == 200:
        st.write("Response from backend:", response.json())
    else:
        st.write("Error:", response.status_code)

