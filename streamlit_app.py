# streamlit_app.py
import streamlit as st
import requests

# URL for FastAPI backend running on localhost inside the Docker container
FASTAPI_URL = "http://localhost:8000/api/data"

st.title("Streamlit App with FastAPI Proxy")

# Button to trigger API call
if st.button("Get Data from FastAPI"):
    try:
        # Make a request to the FastAPI backend
        response = requests.get(FASTAPI_URL)
        data = response.json()
        st.write("Response from FastAPI:", data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling FastAPI: {e}")
