# oi-xgboost-model

A XGBoost Regressor model is trained with around 4400 records of Open interest data of NIFTY index.

## Endpoint 1 : /api/predict

Method allowed : POST

Payload :

```JSON
{
	"day": <integer>,
	"cmp": <float>,
	"calls_change_oi": <integer>,
	"puts_change_oi": <integer>
}
```

___Note : All fields are mandatory___

***

## To run the app

1. Clone the repository

2. CD into the folder

3. Run the below commands
> pip install -r requirements.txt

> uvicorn main:app --reload
