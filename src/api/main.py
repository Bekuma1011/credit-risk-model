from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import mlflow.sklearn
import pandas as pd
from src.feature_engineering import process_data
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API")

# Load the best model from MLflow Model Registry
try:
    model = mlflow.sklearn.load_model("models:/RandomForest_Best_Model/1")  
except Exception as e:
    raise Exception(f"Failed to load model from MLflow: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        data_dict = request.dict()
        input_data = pd.DataFrame([data_dict])
        
        # Process input data using feature engineering pipeline
        processed_data = process_data(input_data)
        
        # Select features for prediction (exclude non-feature columns)
        feature_cols = [col for col in processed_data.columns 
                       if col not in ['TransactionId', 'BatchId', 'AccountId', 
                                     'SubscriptionId', 'CustomerId', 'FraudResult', 
                                     'is_high_risk']]
        X = processed_data[feature_cols]
        
        # Predict risk probability
        prob = model.predict_proba(X)[0][1]  # Probability of high-risk (class 1)
        
        return PredictionResponse(risk_probability=prob)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")