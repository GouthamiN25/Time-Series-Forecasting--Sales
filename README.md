## 📈 Sales Time Series Forecasting — End-to-End Data Science Project

<div align="center">
🔮 ARIMA • SARIMA • Random Forest • XGBoost • LSTM
📊 Full DS Workflow • Feature Engineering • Model Comparison • Deployment Ready
</div>
abc
## 🚀 Project Overview

This project builds an end-to-end Sales Time Series Forecasting system using multiple model families:

Classical Time Series Models

ARIMA

SARIMA

Machine Learning Models

Random Forest

XGBoost

Deep Learning Model

LSTM Sequence Model

This project follows a full Data Science workflow, from EDA → feature engineering → multiple models → evaluation → saving models → deployment using Streamlit/Hugging Face.

## 📂 Project Structure

Time-Series-Forecasting--Sales/
│
├── Time_Series_DS_Project.ipynb 

├── app.py                       

├── requirements.txt              

├── best_rf_model.joblib           

├── time_series_data.csv         

└── README.md   

## 🛠 Tech Stack

| Category      | Tools                                      |
| ------------- | ------------------------------------------ |
| Programming   | Python                                     |
| Data          | Pandas, NumPy                              |
| Modeling      | ARIMA, SARIMA, RandomForest, XGBoost, LSTM |
| Deep Learning | TensorFlow/Keras                           |
| Visualization | Matplotlib, Seaborn                        |
| Deployment    | Streamlit, Hugging Face                    |
| Utils         | joblib, MinMaxScaler                       |


## 🧾 Conclusion

This project presents an end-to-end Sales Time Series Forecasting pipeline built as a complete data science solution, rather than just a single model experiment. Using historical sales data enriched with external factors such as Temperature, Holiday, Promotion, and Store_Visits, the project explores and compares multiple forecasting families:

Classical time series models: ARIMA and SARIMA for capturing trend and seasonality

Machine learning models: Random Forest and XGBoost using lag features and calendar features

Deep learning model: LSTM to learn temporal patterns directly from the sales sequence

The models are evaluated using MAE and RMSE, and their performance is compared to identify the most reliable approach for this dataset. The results show that tree-based models and LSTM can effectively capture non-linear patterns in sales, while ARIMA/SARIMA provide strong statistical baselines and interpretable behavior.

From a user perspective, the final output of this project is:

Forecasted future Sales values for a chosen horizon (e.g., next 7/14/30 days)

Visual plots comparing historical sales vs. model predictions

Saved model artifacts (Random Forest, XGBoost, LSTM, scaler) that are ready to be integrated into a Streamlit app or API for real-time forecasting

Overall, this project demonstrates the full data science lifecycle: from problem framing, EDA, feature engineering, and multi-model experimentation to evaluation and deployment-ready forecasting, making it a strong portfolio piece for real-world sales prediction use cases.

## 👩‍💻 Author

Gouthami Nadupuri

Data Scientist | AI Engineer | Creator of CurioGalaxy 💫

🔗 LinkedIn: https://www.linkedin.com/in/gouthami-nadupuri

🔗 GitHub: https://github.com/GouthamiN25
