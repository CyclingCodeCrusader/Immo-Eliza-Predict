from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a Pydantic model for the expected JSON structure
class SalaryData(BaseModel):
    salary: int
    bonus: int
    taxes: int

# Endpoint using Pydantic model
@app.post("/salary")
async def results(data: SalaryData):
    # The data is automatically validated and parsed into `SalaryData` instance
    if 
    result = data.salary + data.bonus - data.taxes
    return {"message": "Data received successfully", "input": data, "result": result}

@app.get("/")
async def read_root():
    return {"Hello": "World of BeCode ARAI7"}

@app.get("/multi/{value}")
async def multiply(value: int):
    return {"result": value * 2}

@app.post("/results")
async def result(salary, bonus, taxes):
    return {"result": salary + bonus - taxes}