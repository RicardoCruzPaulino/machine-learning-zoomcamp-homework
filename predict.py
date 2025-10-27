
import pickle
from typing import Dict, Any
from fastapi import FastAPI

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

#request model

# Define the model using Pydantic's BaseModel
class Customer(BaseModel):
    """
    Pydantic model for Telco Customer Data, enforcing known categorical values
    and setting constraints on numerical features.
    """
    # =========================================================================
    # 1. DEMOGRAPHICS (Using Literal for strict validation of categories)
    # =========================================================================
    gender: Literal['male', 'female']
    
    # seniorcitizen is 0 or 1, which we treat as an integer with validation
    seniorcitizen: Literal[0, 1]
    
    partner: Literal['yes', 'no']
    dependents: Literal['yes', 'no']

    # =========================================================================
    # 2. SERVICES (Using Literal to enforce valid service options)
    # =========================================================================
    phoneservice: Literal['yes', 'no']
    
    multiplelines: Literal['no', 'yes', 'no_phone_service']
    
    internetservice: Literal['dsl', 'fiber_optic', 'no']
    
    # Internet-dependent services all share the same three categories
    online_security: Literal['no', 'yes', 'no_internet_service']
    online_backup: Literal['no', 'yes', 'no_internet_service']
    device_protection: Literal['no', 'yes', 'no_internet_service']
    tech_support: Literal['no', 'yes', 'no_internet_service']
    streaming_tv: Literal['no', 'yes', 'no_internet_service']
    streaming_movies: Literal['no', 'yes', 'no_internet_service']

    # =========================================================================
    # 3. ACCOUNT AND PAYMENT
    # =========================================================================
    contract: Literal['month-to-month', 'one_year', 'two_year']
    paperless_billing: Literal['yes', 'no']
    
    payment_method: Literal[
        'electronic_check', 
        'mailed_check', 
        'bank_transfer_(automatic)', 
        'credit_card_(automatic)'
    ]

    # =========================================================================
    # 4. NUMERICAL FEATURES (Using Field for validation constraints)
    # =========================================================================
    # tenure must be an integer, and based on the data, it includes 0 to 72.
    tenure: int = Field(ge=0)
    
    # monthlycharges must be a float, and must be non-negative.
    monthly_charges: float = Field(ge=0.0)
    
    # totalcharges must be a float, and must be non-negative.
    total_charges: float = Field(ge=0.0)

#response model

class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="churn-prediction")

with open('model.bin', 'rb') as f_in:
   pipeline = pickle.load(f_in)
 
# X= dv.transform(customer)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.get("/predict")
def predict(customer: Customer) -> PredictResponse:
     
    prob = predict_single(customer.dict())
    
    return PredictResponse (
        churn_probability=prob,
        churn=bool(prob >= 0.5)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)