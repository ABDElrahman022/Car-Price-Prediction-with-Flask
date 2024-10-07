# Car Price Prediction with Flask

This project implements a machine learning model to predict the prices of used cars based on various factors. The application is built using Python with Flask, providing a web interface where users can input car details and receive predicted prices.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Visualization](#visualization)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)

## Overview
The project utilizes a linear regression model from Scikit-Learn to predict car prices. It includes data preprocessing steps, visualization of data distributions and relationships, model building, evaluation, and deployment using Flask.

## Dataset
The dataset used in this project (`cars_data.csv`) contains various features such as mileage, engine volume, brand, body type, engine type, and registration status.

## Data Preprocessing
- Null values were removed as they constituted less than 5% of the data.
- Outliers were handled by applying quantile-based filtering on features like 'Price', 'Mileage', 'EngineV', and 'Year'.
- Log transformation was applied to the target variable 'Price' to normalize its distribution.

## Visualization
- Probability Distribution Functions (PDFs) were used to visualize the distribution of variables and identify outliers.
- Scatter plots were used to check the relationship between 'Price' and other features ('Year', 'EngineV', 'Mileage').

## Model Building
- Features were selected based on their correlation and VIF (Variance Inflation Factor) scores to avoid multicollinearity.
- Categorical features were encoded using dummy variables.
- The dataset was split into training and testing sets, and StandardScaler was applied to scale the inputs.

## Model Evaluation
- The model was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics on both training and testing sets.
- Residual plots were used to check the distribution of errors.

## Deployment
- The Flask framework was used to deploy the machine learning model as a web application.
- The application allows users to input car details and receive real-time price predictions.

## Usage
To run the application locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/ABDElrahman022/Car-Price-Prediction-with-Flask.git
2. Install dependencies:
   ```bash
      pip install -r requirements.txt
3. Run the Flask application:
   ```bash
      python app.py
4. Access the application at http://localhost:5000 in your web browser.

## Future Enhancements
- Improve model accuracy by exploring different machine learning algorithms.
- Enhance the user interface with additional features such as historical price trends and model performance metrics.

