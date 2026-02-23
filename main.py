from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# 1. Load the model
model = joblib.load('irish.pkl')

app = FastAPI()

# 2. incoming data
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Classifier is running!"}

@app.post("/predict")
def predict(data: IrisData):
    # Convert input to the format scikit-learn expects
    features = np.array([[data.sepal_length, data.sepal_width, 
                          data.petal_length, data.petal_width]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Convert numeric prediction (0, 1, 2) back to flower names
    names = ['setosa', 'versicolor', 'virginica']
    return {"prediction": names[int(prediction[0])]}
