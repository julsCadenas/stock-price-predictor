# **📈 Stock Price Prediction with News Sentiment using LSTM**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-stable--improving-yellowgreen.svg)

A deep learning project that predicts Apple's stock prices by combining historical stock data with sentiment analysis of real-world news headlines. Built using TensorFlow/Keras and Hugging Face Transformers.

## **🔍 Overview**
- Objective: Predict the future stock price of Apple Inc. (AAPL) by leveraging both historical price data and news sentiment.
- Stock Data Source: Yahoo Finance
- News Source: New York Times (Apple-related headlines)
- **Sentiment Model:** siebert/sentiment-roberta-large-english
- **ML Model:** LSTM neural network with Keras

## **🧠 Project Workflow**
1. Collect Stock Data
    - Downloaded AAPL historical price data from Yahoo Finance.
2. Collect News Headlines
    - Scraped or queried Apple-related headlines from the New York Times.
3. Sentiment Analysis
    - Used a pre-trained Roberta model from Hugging Face to classify each headline as positive, negative, or neutral.
    - Aggregated daily sentiment scores.
    - Feature Engineering
4. Merged sentiment scores with stock price features (e.g., Close, Volume).
    - Scaled features using MinMaxScaler.
5. Model Training
    - Built and trained an LSTM model using TensorFlow/Keras.
    - Trained on sequences of combined price + sentiment data.

## **📊 Evaluation Results**

| Metric               | Train  | Validation | Test   |
| -------------------- | ------ | ---------- | ------ |
| **RMSE**             | 4.95   | 7.81       | 6.88   |
| **MAE**              | 3.99   | 6.12       | 5.48   |
| **R² Score**         | 0.9228 | 0.8870     | 0.8471 |
| **MAPE**             | 2.49%  | 2.98%      | 2.51%  |
| **Directional Acc.** | 53.09% | 45.89%     | 51.03% |

### **Actual vs. Predicted Stock Prices (Test Set)**

![Actual vs. Predicted Stock Prices](misc/output.png)

## **🧠 Current Model Architecture**
The model was built using TensorFlow/Keras and is designed to capture sequential dependencies in the combined stock price and news sentiment time series data.
```python
model = Sequential([
    Input(shape=input_shape),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

optimizer = Adam(learning_rate=0.0003)
model.compile(optimizer=optimizer, loss=Huber(delta=1.0))
```
- Input Layer: Takes sequences of historical stock and sentiment features
- LSTM Layer: 32 units to capture temporal dependencies
- BatchNormalization: Helps stabilize and accelerate training
- Dense Layer: 16 neurons with ReLU activation
- Dropout Layer: 30% dropout rate for regularization
- Output Layer: Single neuron predicting the next stock price
- Loss Function: Huber Loss (delta=1.0) — more robust to outliers than MSE
- Optimizer: Adam with a learning rate of 0.0003

## **🚀 Future Improvements**
- Improve directional accuracy via classification or hybrid models
- Experiment with other sentiment models (FinBERT, financial BERT variants)
- Add more features like RSI, MACD, and volume spikes
- Build an interactive dashboard to visualize predictions

## **⚖️ License**

This project is licensed under the [MIT License](LICENSE).
