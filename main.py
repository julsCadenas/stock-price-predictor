import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

st.set_page_config(page_title="Stock Price Predictor with News Sentiment", layout="wide")

# Load model and artifacts
@st.cache_data
def load_model_and_artifacts():
    model = load_model('./notebooks/gru_output/gru_stock_model.keras')
    feature_scaler = joblib.load('./notebooks/artifacts/feature_scaler.pkl')
    target_scaler = joblib.load('./notebooks/artifacts/target_scaler.pkl')
    X = np.load('./notebooks/artifacts/final_X.npy')
    y = np.load('./notebooks/artifacts/final_y.npy')
    df = pd.read_csv('./data/final_merged.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return model, feature_scaler, target_scaler, X, y, df

model, feature_scaler, target_scaler, X, y, df = load_model_and_artifacts()

st.title("Stock Price Predictor with News Sentiment")
st.markdown("Predicts stock closing prices using GRU model trained on historical price + news sentiment.")

# Map sequences to dates
sequence_length = 30
dates_for_sequences = df['Date'].iloc[sequence_length:].reset_index(drop=True)

# ===========================
# Slider
# ===========================
st.subheader("Select Range of Date Samples to Display")
test_start = len(X) - 195  # start of test set
test_end = len(X) - 1      # end of test set

# Convert sequence dates to datetime.date
sequence_dates = dates_for_sequences.dt.date

# slider proper
sample_range = st.slider(
    "Test Sample Range",
    min_value=sequence_dates.iloc[test_start],
    max_value=sequence_dates.iloc[test_end],
    value=(sequence_dates.iloc[test_end-50], sequence_dates.iloc[test_end])
)

# get indices
# Convert to numpy array for searchsorted
sequence_dates_np = sequence_dates.to_numpy()

# Find the closest indices for slider selection
start_idx = np.searchsorted(sequence_dates_np, sample_range[0], side="left")
end_idx = np.searchsorted(sequence_dates_np, sample_range[1], side="right") - 1

# Ensure indices are within bounds
start_idx = max(0, min(start_idx, len(sequence_dates_np) - 1))
end_idx = max(0, min(end_idx, len(sequence_dates_np) - 1))

X_sample = X[start_idx:end_idx + 1]
y_sample = y[start_idx:end_idx + 1]
selected_dates = dates_for_sequences[start_idx:end_idx + 1]

# ===========================
# Make Predictions
# ===========================
y_pred_scaled = model.predict(X_sample)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual = target_scaler.inverse_transform(y_sample)

# ===========================
# Display Metrics & Chart
# ===========================
mae = np.mean(np.abs(y_actual - y_pred))
st.metric(label="Mean Absolute Error (selected range)", value=f"{mae:.2f}") # Display MAE

st.subheader(f"Predicted vs Actual Close Prices ({selected_dates.iloc[0].date()} to {selected_dates.iloc[-1].date()})")
# Line chart
df_plot = pd.DataFrame({
    "Actual": y_actual.flatten(),
    "Predicted": y_pred.flatten()
}, index=selected_dates)
st.line_chart(df_plot)

# Show data table
if st.checkbox("Show Data Table"):
    st.dataframe(df_plot)

st.success("Predictions complete!")
