import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# ---- Load data and model ----
@st.cache_data
def load_data():
    df = pd.read_csv("time_series_data.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("best_rf_model.joblib")
    return model

df = load_data()
model = load_model()

# ---- Helper: create lag features (must match notebook) ----
def create_lag_features(data, target_col="Sales", lags=7):
    df_ml = data.copy()
    for lag in range(1, lags + 1):
        df_ml[f"lag_{lag}"] = df_ml[target_col].shift(lag)
    df_ml["dayofweek"] = df_ml.index.dayofweek
    df_ml["month"] = df_ml.index.month
    df_ml["dayofmonth"] = df_ml.index.day
    df_ml = df_ml.dropna()
    return df_ml

feature_cols = [c for c in create_lag_features(df).columns if c != "Sales"]

def predict_next_n_days_rf(model, last_df, n_days=7):
    preds = []
    tmp = last_df.copy()
    for _ in range(n_days):
        tmp_ml = create_lag_features(tmp)[feature_cols]
        x_last = tmp_ml.iloc[[-1]]
        y_hat = model.predict(x_last)[0]
        next_index = tmp.index[-1] + timedelta(days=1)
        new_row = tmp.iloc[[-1]].copy()
        new_row.name = next_index
        new_row["Sales"] = y_hat
        tmp = pd.concat([tmp, new_row])
        preds.append((next_index, y_hat))
    return pd.DataFrame(preds, columns=["Date", "Predicted_Sales"]).set_index("Date")

# ---- Streamlit UI ----
st.title("Sales Time Series Forecasting")
st.write("This app uses a Random Forest model trained on time-series features to forecast future Sales.")

st.subheader("Historical Sales")
st.line_chart(df["Sales"])

horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)

if st.button("Run Forecast"):
    future_df = predict_next_n_days_rf(model, df[["Sales", "Temperature", "Holiday", "Promotion", "Store_Visits"]], n_days=horizon)
    st.subheader("Forecasted Sales")
    st.dataframe(future_df)

    # Plot history + forecast
    fig, ax = plt.subplots()
    df["Sales"].tail(100).plot(ax=ax, label="History")
    future_df["Predicted_Sales"].plot(ax=ax, label="Forecast")
    ax.legend()
    st.pyplot(fig)
