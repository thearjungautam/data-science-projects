# House Price Prediction and Home Quality Classification

## Overview

This project analyzes over 21,000 residential home sales in King County, Washington to identify the factors that influence housing prices and home quality. Using statistical modeling and machine learning techniques, the project develops both regression and classification models to support data-driven real estate insights.

## Objectives

- Predict home sale prices using property characteristics and location features
- Identify key drivers of housing value
- Classify whether a home is considered "good quality"
- Evaluate model performance using statistical and predictive metrics

## Dataset

- 21,613 home sales from King County, Washington
- Sales period: May 2014 – May 2015
- Features include:
  - Bedrooms
  - Bathrooms
  - Square footage
  - Waterfront status
  - View rating
  - Condition
  - Construction grade
  - Year built
  - Location information

## Data Preparation

- Corrected data quality issues and invalid records
- Removed unrealistic observations and missing values
- Created engineered features including:
  - Metro area indicator
  - Renovation indicator
  - Basement indicator
  - Seasonal housing market variables
  - Home quality classification target

## Methods

### Price Prediction

Developed multiple linear regression models to predict housing prices:

- Feature selection and model refinement
- Multicollinearity analysis using VIF
- Box-Cox transformation analysis
- Log-transformed regression modeling
- Assumption testing and residual diagnostics

### Home Quality Classification

Built logistic regression models to predict whether a home is classified as high quality based on condition and construction grade.

- Variable selection
- Likelihood ratio testing
- ROC analysis
- Threshold optimization
- Confusion matrix evaluation

## Results

### Housing Price Model

- Final model achieved an Adjusted R² of approximately **0.66**
- Significant predictors included:
  - Waterfront access
  - Construction grade
  - Living space
  - View quality
  - Metro area location

### Home Quality Classification

- ROC-AUC: **0.95**
- Classification accuracy: **89.5%**
- Improved recall through threshold tuning

## Technologies

- R
- tidymodels
- caret
- ggplot2
- Statistical Modeling
- Linear Regression
- Logistic Regression
- Data Visualization

## Key Learning Outcomes

- Predictive modeling
- Feature engineering
- Model diagnostics
- Classification analysis
- Statistical inference
- Data cleaning and validation
- Threshold optimization

## Business Impact

This project demonstrates how statistical learning methods can be applied to real estate data to estimate housing values, identify high-quality properties, and support data-driven decision-making for buyers, sellers, and market analysts.
