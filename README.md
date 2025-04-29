# Advanced Logistic Regression Classifier

This is an interactive web application that demonstrates binary classification using Logistic Regression with a modern and attractive GUI interface.

## Features

- Interactive web interface built with Streamlit
- Multiple dataset options (Breast Cancer, Iris, and custom datasets)
- Real-time model parameter tuning
- Comprehensive model evaluation metrics
- Interactive visualizations including:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance
  - Sigmoid Function
- Support for custom dataset upload

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, execute:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Select a dataset from the sidebar (Breast Cancer, Iris, or upload your own)
2. Adjust model parameters:
   - Regularization Strength (C)
   - Maximum Iterations
   - Test Size
3. View the model performance metrics and visualizations
4. For custom datasets, upload a CSV file with your data

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies 