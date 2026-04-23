import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Create directories to store models and outputs
os.makedirs("models", exist_ok=True)
DATA_PATH = "dataset/delhi_pm25_aqi.csv"

def prepare_data():
    """Loads dataset, handles missing values, and creates unvariate lag features."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing: Convert to datetime and rename columns
    df = df.rename(columns={'value': 'pm25', 'period.datetimeFrom.utc': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Resample to hourly and fill missing (forward fill)
    df = df['pm25'].resample('H').mean().ffill().reset_index()
    
    # Feature Engineering: Create lag features for time-series forecasting (T-1, T-2, T-3)
    df['lag_1'] = df['pm25'].shift(1)
    df['lag_2'] = df['pm25'].shift(2)
    df['lag_3'] = df['pm25'].shift(3)
    
    df.dropna(inplace=True)
    return df

def plot_predictions(y_test, preds, model_name):
    """Bonus: Plots Actual vs Predicted values."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Actual PM2.5", color='blue')  # First 100 samples
    plt.plot(preds[:100], label=f"Predicted PM2.5 ({model_name})", color='red')
    plt.title(f"{model_name}: Actual vs Predicted PM2.5 (First 100 Hours)")
    plt.xlabel("Time Steps")
    plt.ylabel("PM2.5 Level")
    plt.legend()
    plt.savefig(f"models/{model_name}_predictions.png")
    plt.close()

def train_svr(X_train, X_test, y_train, y_test):
    """Trains a Support Vector Regressor (SVR) model."""
    print("Training SVR (RBF Kernel)...")
    svr = SVR(kernel='rbf', C=100, gamma='scale')
    svr.fit(X_train, y_train)
    
    preds = svr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"SVR -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
    plot_predictions(y_test, preds, "SVR")
    return svr, rmse, r2

def train_lstm(X_train, X_test, y_train, y_test):
    """Trains a Deep Learning LSTM model."""
    print("Training LSTM...")
    # Reshape for LSTM: (samples, timesteps, features)
    X_train_lstm = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=10, verbose=0, batch_size=32)
    
    preds = model.predict(X_test_lstm, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"LSTM -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
    plot_predictions(y_test, preds, "LSTM")
    return model, rmse, r2

if __name__ == "__main__":
    df = prepare_data()
    features = ['lag_1', 'lag_2', 'lag_3']
    target = 'pm25'
    
    X = df[features]
    y = df[target]
    
    # Normalize/Scale features (Crucial for distance-based SVR and LSTM)
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Train/Test Split (Sequential for time series)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    
    # MLflow tracking
    mlflow.set_experiment("Air_Quality_SVR_LSTM_Pipeline")
    
    with mlflow.start_run(run_name="SVR"):
        svr_model, svr_rmse, svr_r2 = train_svr(X_train, X_test, y_train, y_test)
        mlflow.log_metric("rmse", svr_rmse)
        mlflow.log_metric("r2", svr_r2)
        mlflow.sklearn.log_model(svr_model, "svr_model")

    with mlflow.start_run(run_name="LSTM"):
        lstm_model, lstm_rmse, lstm_r2 = train_lstm(X_train, X_test, y_train, y_test)
        mlflow.log_metric("rmse", lstm_rmse)
        mlflow.log_metric("r2", lstm_r2)

    # Clean old models
    if os.path.exists("models/best_model.pkl"): os.remove("models/best_model.pkl")
    if os.path.exists("models/best_model.keras"): os.remove("models/best_model.keras")

    # Comparison and Best Model Selection
    print("\n--- MODEL COMPARISON ---")
    print(f"SVR R2 Score:  {svr_r2:.4f} | RMSE: {svr_rmse:.4f}")
    print(f"LSTM R2 Score: {lstm_r2:.4f} | RMSE: {lstm_rmse:.4f}")

    if svr_r2 >= lstm_r2: 
        print("⭐⭐ SVR is the Best Model. Saving...")
        joblib.dump(svr_model, "models/best_model.pkl")
    else:
        print("⭐⭐ LSTM is the Best Model. Saving...")
        lstm_model.save("models/best_model.keras")
        
    joblib.dump(scaler_X, "models/scaler.pkl")
    print("Pipeline Execution Finished Successfully. Plots saved in 'models/' directory.")
