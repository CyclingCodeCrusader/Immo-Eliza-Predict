from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
 
# Creating FastAPI instance
app = FastAPI()
 
@app.get("/")
async def root():
    """Route that returns 'Alive!' if the server runs."""
    return {"Status": "Alive! Welcome ! This is local from VSCODE and FastApi."}

@app.get('/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to testapi!, {name}'}

@app.post('/hello/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to testapi via POST!, {name}'}

# Define a Pydantic model for the expected JSON structure
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

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
 
# Loading Iris Dataset
iris = load_iris()
 
# Getting our Features and Targets
X = iris.data
Y = iris.target
 
# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X,Y)

# HIER KAN DUS HET MODEL INGELADEN WORDEN VAN OP EEN JOBLIB FILE EN IN EEN VARIABELE GESTOEKEN WORDEN
# BV VARIABELE MODEL =

# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
     
    # Predicting the Class
    class_idx = clf.predict(test_data)[0]
     
    # Return the Result
    return { 'class' : iris.target_names[class_idx]}


# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
 
# Loading Iris Dataset
iris = load_iris()
 
# Getting our Features and Targets
X = iris.data
Y = iris.target
 
# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X,Y)

# HIER KAN DUS HET MODEL INGELADEN WORDEN VAN OP EEN JOBLIB FILE EN IN EEN VARIABELE GESTOEKEN WORDEN
# BV VARIABELE MODEL =

# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
     
    # Predicting the Class
    class_idx = clf.predict(test_data)[0]
     
    # Return the Result
    return { 'class' : iris.target_names[class_idx]}