import numpy as np
from fastapi import FastAPI
from xgboost import XGBRegressor
from pydantic import BaseModel

# load the XGboost model
model = XGBRegressor()
model.load_model("model/xgboost_model_v1.json")

app = FastAPI()


class ModelInput(BaseModel):
    day: int
    cmp: float
    calls_change_oi: int
    puts_change_oi: int


@app.get("/")
async def main():
    return {'Status':'OK', 'Message': 'API is running...'}


@app.post("/api/predict")
async def model_predict(payload: ModelInput):

    print(payload)

    # prepare numpy array with shape (4, 1) to feed the model
    x_input = np.array([
        [payload.day, payload.cmp, payload.calls_change_oi, payload.puts_change_oi]
    ])
    
    try:
        # model prediction
        y_pred = model.predict(x_input)
        y_pred = round(y_pred[0], 2)
        print("Prediction is :", y_pred)

        return {
            'Status':'OK',
            'Message': 'Prediction successfull',
            'PredictedValue': float(y_pred)
        }
    
    except Exception as e:

        return {
            'Status':'Failed',
            'Message': f'Error occured : {e}'
        }