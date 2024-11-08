from fastapi import FastAPI
import os

# Set port to the env variable PORT to make it easy to choose the port on the server
# If the Port env variable is not set, use port 8000
PORT = os.environ.get("PORT", 8000)
app = FastAPI(port=PORT)


@app.get("/")
async def root():
    """Route that returns 'Alive!' if the server runs."""
    return {"Status": "Alive! Welcome !"}

@app.get("/multi/{value}")
async def multiply(value: int):
    return {"result": value * 3}
# Defining path operation for /name endpoint

@app.get('/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to GeeksforGeeks!, {name}'}

@app.get("/hello")
async def say_hello(user: str = "Anonymous"):
    """Route that will return 'hello {user}'."""
    return {"Message": f"Hello {user}!"}

# Define a Pydantic model for the expected JSON structure
from pydantic import BaseModel
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



import streamlit as st
import time

# Sidebar
st.sidebar.title("Immo Eliza")
st.sidebar.image("assets\BeCode Harmony Real Estate Logo.jpg", width=100)

st.sidebar.subheader('Web Scraper')
st.sidebar.subheader('Data Cleaner')
st.sidebar.subheader('Data Analyzer')
st.sidebar.subheader('Modeller')
st.sidebar.subheader('Predictor')

# Input parameters
container1 = st.container()
col1,col2, col3=container1.columns([1,1,1])
#col2.image("assets\house.jpg", width = 200)
col2.subheader("Predictor")


