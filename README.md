# California House Price Prediction API

This project is a Machine Learning API built using FastAPI. It predicts California house prices using a trained Random Forest Regression model.

## How It Works

### Step 1: Train the Model

The model is trained using the California Housing Dataset from Scikit-Learn.

Training process:

* Load the dataset
* Split data into training and testing sets
* Train a Random Forest Regressor
* Evaluate model performance
* Save the trained model (`house_model.joblib`)
* Save feature names (`house_features.joblib`)

---

### Step 2: Load the Model

When the FastAPI server starts:

* The trained model is loaded from `house_model.joblib`
* The feature list is loaded from `house_features.joblib`

---

### Step 3: Single House Prediction

The `/predict` endpoint accepts house details as JSON input.

Example:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.984,
  "AveBedrms": 1.024,
  "Population": 322,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

Workflow:

User Input → Data Validation → Convert to DataFrame → Model Prediction → Return House Price

---

### Step 4: Bulk Prediction Using CSV

The `/predict-file` endpoint allows users to upload a CSV file containing multiple house records.

Workflow:

Upload CSV → Validate Columns → Read File Using Pandas → Model Prediction → Add Predicted Price Column → Download Updated CSV

---

## API Endpoints

### GET /

Returns API status.

### GET /health

Returns model and API health information.

### POST /predict

Predicts the price of a single house.

### POST /predict-file

Predicts house prices for multiple records from a CSV file.

---

## Technologies Used

* FastAPI
* Python
* Pandas
* Scikit-Learn
* Pydantic
* Joblib

## Project Workflow

Dataset → Model Training → Save Model → FastAPI → User Input → Prediction → Result
