# Stock Sentiment Predictor - Starter Code
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockSentimentPredictor:
    def __init__(self, ticker, newsapi_key):
        self.ticker = ticker
        self.newsapi = NewsApiClient(api_key=newsapi_key)
        self.scaler = MinMaxScaler()
        self.model = None
        
    def get_stock_data(self, start_date, end_date):
        """Fetch stock data and calculate technical indicators"""
        print(f"Fetching stock data for {self.ticker}...")
        
        # Get stock data
        stock_data = yf.download(self.ticker, start=start_date, end=end_date)
        
        # Calculate technical indicators
        stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['Volatility'] = stock_data['Close'].rolling(window=20).std()
        stock_data['Volume_MA'] = stock_data['Volume'].rolling(window=20).mean()
        
        return stock_data
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_news_data(self, start_date, end_date):
        """Fetch news data and calculate sentiment"""
        print(f"Fetching news data for {self.ticker}...")
        
        # Get news headlines
        news_data = self.newsapi.get_everything(
            q=self.ticker,
            from_param=start_date,
            to=end_date,
            language='en',
            sort_by='publishedAt'
        )
        
        # Process news data
        news_df = pd.DataFrame(news_data['articles'])
        
        if len(news_df) == 0:
            print("No news data found!")
            return pd.DataFrame()
        
        # Calculate sentiment scores
        news_df['sentiment'] = news_df['title'].apply(self.get_sentiment)
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
        news_df['date'] = news_df['publishedAt'].dt.date
        
        # Aggregate daily sentiment
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment': ['mean', 'std', 'count']
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_volatility', 'news_count']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        return daily_sentiment
    
    def get_sentiment(self, text):
        """Calculate sentiment score using TextBlob"""
        if pd.isna(text):
            return 0
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def prepare_features(self, stock_data, news_data):
        """Combine stock and news data into features"""
        print("Preparing features...")

        # Ensure stock_data has a single-level column index and a 'date' column
        stock_data = stock_data.copy()
        if 'Date' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['Date'])
            stock_data = stock_data.drop(columns=['Date'])
        elif stock_data.index.name == 'Date':
            stock_data = stock_data.reset_index()
            stock_data['date'] = pd.to_datetime(stock_data['Date'])
            stock_data = stock_data.drop(columns=['Date'])
        else:
            stock_data = stock_data.reset_index()
            stock_data['date'] = pd.to_datetime(stock_data['index'])
            stock_data = stock_data.drop(columns=['index'])

        # Ensure news_data has a 'date' column and is not a MultiIndex
        if len(news_data) > 0:
            if isinstance(news_data.columns, pd.MultiIndex):
                news_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in news_data.columns.values]
            if not pd.api.types.is_datetime64_any_dtype(news_data['date']):
                news_data['date'] = pd.to_datetime(news_data['date'])
            merged_data = pd.merge(stock_data, news_data, on='date', how='left')
            # Fill missing sentiment values
            merged_data['avg_sentiment'] = merged_data['avg_sentiment'].fillna(0)
            merged_data['sentiment_volatility'] = merged_data['sentiment_volatility'].fillna(0)
            merged_data['news_count'] = merged_data['news_count'].fillna(0)
        else:
            merged_data = stock_data.copy()
            merged_data['avg_sentiment'] = 0
            merged_data['sentiment_volatility'] = 0
            merged_data['news_count'] = 0

        # Select features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MA_20', 'MA_50', 'Volatility', 'Volume_MA',
            'avg_sentiment', 'sentiment_volatility', 'news_count'
        ]

        # Remove rows with NaN values
        merged_data = merged_data.dropna()

        # Create target variable (next day's closing price)
        merged_data['target'] = merged_data['Close'].shift(-1)
        merged_data = merged_data.dropna()

        return merged_data[feature_columns + ['target']]
    
    def create_sequences(self, data, sequence_length=60):
        """Create sequences for LSTM model"""
        print("Creating sequences...")
        
        features = data.drop('target', axis=1)
        targets = data['target']
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(targets.iloc[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, start_date, end_date, sequence_length=60):
        """Train the model"""
        # Get data
        stock_data = self.get_stock_data(start_date, end_date)
        news_data = self.get_news_data(start_date, end_date)
        
        # Prepare features
        features = self.prepare_features(stock_data, news_data)
        
        # Create sequences
        X, y = self.create_sequences(features, sequence_length)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build and train model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history
    
    def predict(self, recent_data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare recent data
        recent_scaled = self.scaler.transform(recent_data)
        recent_sequence = recent_scaled[-60:].reshape(1, 60, recent_scaled.shape[1])
        
        # Make prediction
        prediction = self.model.predict(recent_sequence)
        return prediction[0][0]

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = StockSentimentPredictor(
        ticker="AAPL",
        newsapi_key="e8da37ab42f2481aad4a8cec18db51c4"  # Replace with your NewsAPI key
    )
    
    # Train model
    history = predictor.train(
        start_date="2025-06-17",
        end_date="2025-07-10"
    )
    
    print("Model training completed!")
    print("Next steps:")
    print("1. Save the model: predictor.model.save('stock_model.h5')")
    print("2. Test predictions on recent data")
    print("3. Build a web interface or API")