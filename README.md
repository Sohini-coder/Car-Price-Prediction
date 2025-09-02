# Car Price Prediction using Machine Learning  

## Project Overview  
This project is part of PCC-CS881 (College Project).  
The objective is to predict used car prices based on multiple features such as make, model, year, mileage, engine specifications, and other attributes.  

We experimented with several machine learning models and finalized the Random Forest Regressor as the best-performing model.  
A Streamlit web application is also deployed for interactive car price predictions.  

---

## Features  
- Data preprocessing & cleaning (missing values, encoding, scaling).  
- Comparison of multiple ML algorithms:  
  - Linear Regression  
  - Lasso Regression  
  - Ridge Regression
  - Support Vector Regression (SVR)
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - XGBoost 
- Final Model: Random Forest Regressor.  
- Evaluation metrics: R², MAE, RMSE.  
- Streamlit app with:  
  - Input form for 87 features.  
  - Feature importance visualization.  
  - Predicted car price output.  

---

## Dataset  
- Dataset contains car details with 87 input features.  
- Preprocessing steps:  
  - Handling categorical & numerical features.  
  - Feature engineering.  
  - Duplicate removal.  
  - Train-test split.  

---

##  Model Performance  
| Model                | R² Score | MAE     | RMSE    |  
|-----------------------|---------|---------|---------|  
| Linear Regression     | 0.5549  | 187,518 | 339,783 |  
| Lasso Regression      | 0.5561  | 186,595 | 339,282 |  
| Ridge Regression      | 0.5750  | 189,112 | 332,229 |  
| SVR                   | -0.1798 | 241,781 | 555,847 |
| Decision Tree         | 0.9014  | 69,003  | 158,824 |  
| Random Forest (Final) | 0.9575  | 49,621  | 108,692 |  
| XGBoost               | 0.9506  | 55,405  | 111,811 |  

---

## Deployment  
Deployed using Streamlit.  

Features of the app:

- User-friendly interface.

- Real-time predictions.

- Feature importance chart.



