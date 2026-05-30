# Predicting Food Recall Risk and Public Health Costs

## Overview

This project develops a machine learning framework to predict food recall risk and estimate associated public health costs using environmental, geographic, and temporal data. By combining FDA food recall records with NOAA weather data, the project investigates whether environmental conditions can help identify periods of elevated pathogen-related risk.

The project uses a two-stage modeling approach that separates event occurrence from event severity, reflecting the different factors that influence recall frequency and economic impact.

## Business Problem

Public health agencies often react to foodborne illness outbreaks after they occur because predicting pathogen-related events is challenging. The objective of this project is to develop an early-warning system that can identify periods and locations with elevated recall risk, allowing for more proactive resource allocation and inspection planning.

## Dataset

Data were collected from multiple public sources:

- FDA Food Enforcement Reports (openFDA)
- NOAA National Climatic Data Center (nClimDiv)
- USDA Economic Research Service (ERS)

The final dataset combines:

- State-level weather observations
- Food recall events
- Pathogen classifications extracted from recall descriptions
- Economic cost estimates for foodborne illnesses
- Geographic and temporal indicators

## Feature Engineering

Features were engineered across four categories:

### State Features
- Historical recall frequency
- Baseline state-level risk
- Average pathogen cost indicators

### Event History Features
- Prior recall occurrence
- Rolling event counts
- Recent recall activity

### Temporal Features
- Month
- Seasonal effects
- Cyclical month encoding (sine/cosine)

### Weather Features
- Temperature
- Precipitation
- Lagged weather variables
- Rolling averages
- Weather anomalies
- Temperature-precipitation interaction terms

## Methodology

### Stage 1: Recall Event Prediction

Built XGBoost classification models to predict whether a recall event would occur in a given state-month.

Models evaluated:

1. Weather-only model
2. State + temporal + event-history model
3. Full model (all features)

Evaluation metrics:

- ROC-AUC
- Average Precision
- Precision
- Recall
- F1 Score

### Stage 2: Cost Severity Prediction

Built XGBoost regression models to estimate pathogen-related economic costs conditional on an event occurring.

Evaluation metrics:

- RMSE
- MAE
- R²

## Results

### Recall Event Classification

The strongest model achieved:

- ROC-AUC ≈ 0.99
- Average Precision ≈ 0.86
- Recall ≈ 0.91
- F1 Score ≈ 0.80

Key finding:

State-level and temporal patterns were far more predictive than weather variables alone.

### Cost Severity Regression

The severity model showed limited predictive power:

- R² ≈ 0.03
- MAE ≈ 2.75

Key finding:

Environmental variables contain some signal for event occurrence but explain very little variation in economic cost outcomes.

## Key Insights

- Recall events follow learnable geographic and temporal patterns.
- State-level risk factors dominate predictive performance.
- Weather variables provide only modest incremental value.
- Cost severity appears to be driven by factors not captured in environmental datasets, such as supply chain dynamics and contamination scale.
- Classification models are substantially more useful than cost forecasting models for operational decision-making.

## Technologies

- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- NOAA Climate Data
- FDA Open Data APIs

## Skills Demonstrated

- Machine Learning
- Classification Modeling
- Regression Modeling
- Feature Engineering
- Time-Series Cross Validation
- Public Health Analytics
- Data Integration
- API Data Collection
- Predictive Modeling
- Model Evaluation

## Applications

This project demonstrates how machine learning can support public health agencies by providing early-warning systems for food safety risk, helping prioritize inspections, allocate resources, and monitor emerging pathogen threats.
