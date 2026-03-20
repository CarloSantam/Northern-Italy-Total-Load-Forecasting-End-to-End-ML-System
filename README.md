# Northern Italy Total Load Forecasting – End-to-End ML System

## Overview
This project implements an end-to-end machine learning pipeline for forecasting electricity demand (total load) in Northern Italy.

It integrates:
- ENTSO-E historical and forecast load data
- Copernicus atmospheric and weather forecasts

The system includes:
- Data ingestion and preprocessing
- Feature engineering (lag, calendar, weather features)
- Model training and evaluation
- Probabilistic forecasting (quantile models)
- Model explainability (SHAP)
- Interactive dashboard (Streamlit + LLM assistant)

## Objectives
- Build a robust and reproducible load forecasting pipeline
- Improve forecast accuracy using exogenous weather data
- Compare multiple ML models for probabilistic forecasting
- Provide interpretability and transparency of predictions
- Deliver a deployable and user-friendly interface

## Project Structure
.
├── Data/
│   ├── Weather Data/
├── Forecast.py
├── Streamlit_app.py
├── Weather Data Download.py
├── requirements.txt
└── README.md

## Data Sources
- ENTSO-E: electricity load data
- Copernicus: weather forecasts (temperature, wind, etc.)

These variables are critical because electricity demand is strongly influenced by weather and temporal patterns.

## Feature Engineering
The following features are generated:
- Lag features (past load values)
- Rolling statistics
- Calendar features (hour, weekday, seasonality)
- Weather features (temperature, wind, etc.)

## Models
The project benchmarks several machine learning models:

- XGBoost
- CatBoost
- Quantile Random Forest (QRF)

These models are used for probabilistic forecasting (quantile prediction).

## Evaluation Metrics
- WAPE (Weighted Absolute Percentage Error)
- Additional regression metrics where relevant

Achieved performance:
- ~5% WAPE on 2025 backtest

## Explainability
Model predictions are interpreted using SHAP:
- Feature importance
- Local explanations
- Model transparency

## Streamlit Dashboard
A local dashboard is implemented to:
- Visualize forecasts
- Explore model outputs
- Interact with an LLM assistant

To run:

streamlit run Streamlit_app.py

## Installation

Clone the repository:

git clone https://github.com/CarloSantam/Northern-Italy-Total-Load-Forecasting-End-to-End-ML-System.git
cd Northern-Italy-Total-Load-Forecasting-End-to-End-ML-System


Install dependencies:

pip install -r requirements.txt

## Usage

1. Download weather data:
python "Weather Data Download.py"

2. Run forecasting pipeline:
python Forecast.py

3. Launch dashboard:
streamlit run Streamlit_app.py

## Key Features
- End-to-end ML pipeline
- Integration of energy + weather data
- Probabilistic forecasting
- Model comparison framework
- Explainability with SHAP
- Interactive dashboard

## Future Improvements
- Add deep learning models (LSTM, Transformers)
- Deploy on cloud (AWS / GCP)
- Real-time data ingestion
- Automated retraining (MLOps)

## License
This project is licensed under the MIT License.

## Author
Carlo Santambrogio
