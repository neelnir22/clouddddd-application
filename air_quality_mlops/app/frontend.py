import streamlit as st
import requests
import os
import pandas as pd
import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "delhi_pm25_aqi.csv")

st.set_page_config(page_title="PM2.5 Time-Series Prediction", layout="wide")

@st.cache_data
def load_historical_data():
    """Loads and caches the physical dataset for the Date Lookup feature."""
    if not os.path.exists(DATASET_PATH): 
        return None
    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns={'value': 'pm25', 'period.datetimeFrom.utc': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df.set_index('datetime', inplace=True)
    # Ensure continuous hourly data exactly like the training script
    df = df['pm25'].resample('H').mean().ffill()
    return df

st.title("🏙️ Delhi PM2.5 Hourly Forecast System")

data = load_historical_data()

# Provide the user a choice between looking up actual old dates or inputting manually
mode = st.radio("⚙️ Select Prediction Mode:", ["Historical Date Lookup (Interactive)", "Manual Custom Forecasting"], horizontal=True)
st.markdown("---")

st.sidebar.markdown("### 📊 PM2.5 AQI Guidelines")
aqi_data = {
    "Category": ["Good 🟢", "Satisfactory 🟡", "Moderately Polluted 🟠", "Poor 🔴", "Very Poor 🟣", "Severe 🟫"],
    "Range (µg/m³)": ["0 - 30", "31 - 60", "61 - 90", "91 - 150", "151 - 250", "251+"]
}
# Index is hidden by dataframe naturally but standard table renders index, hide it
st.sidebar.dataframe(pd.DataFrame(aqi_data), hide_index=True, use_container_width=True)
st.sidebar.markdown("---")

pm25_lag_1, pm25_lag_2, pm25_lag_3 = 170.0, 160.0, 150.0 # Default safe fallback
can_predict = True
forecast_time_label = "Next Hour"

if mode == "Historical Date Lookup (Interactive)" and data is not None:
    st.markdown("### 📅 Browse Historical Data & Back-Test The Model")
    st.markdown("Select a date and time from the past. The app will fetch the actual pollution level at that moment, and ask the Machine Learning model to forecast the *following* hour.")
    
    min_date, max_date = data.index.min().date(), data.index.max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        sel_date = st.date_input("Select a Historical Date", min_value=min_date, max_value=max_date, value=max_date - datetime.timedelta(days=2))
    with col2:
        # Generate generic 24 hours selection
        sel_time = st.selectbox("Select Hour of the Day", [datetime.time(i, 0) for i in range(24)])
        
    target_dt = pd.to_datetime(f"{sel_date} {sel_time}")
    
    try:
        # Extract the target time + the previous hours required for the Lag inputs
        t_0 = data.loc[target_dt]                                     # Selected Time
        t_m1 = data.loc[target_dt - pd.Timedelta(hours=1)]            # T-1 Hour
        t_m2 = data.loc[target_dt - pd.Timedelta(hours=2)]            # T-2 Hours
        # Note: We use t_0, t_m1, t_m2 into the model to predict T+1
        
        pm25_lag_1 = t_0   # Most recent data point to feed model is the selected time
        pm25_lag_2 = t_m1
        pm25_lag_3 = t_m2
        
        forecast_dt = target_dt + pd.Timedelta(hours=1)
        forecast_time_label = forecast_dt.strftime('%Y-%m-%d %H:00')
        
        st.info(f"📍 **Actual Recorded Data for {target_dt.strftime('%Y-%m-%d %H:00')}**  \nThe real PM2.5 level was heavily recorded at: **{t_0:.2f} µg/m³**")
        
        with st.expander("View internal data feeding the model"):
            st.write(f"- Parameter `lag_3`: {t_m2:.2f} µg/m³")
            st.write(f"- Parameter `lag_2`: {t_m1:.2f} µg/m³")
            st.write(f"- Parameter `lag_1`: {t_0:.2f} µg/m³")
            st.write(f"-> **Target Prediction Time:** {forecast_time_label}")
            
    except KeyError:
        st.warning("⚠️ Continuous historical data is missing for this exact 3-hour window in the dataset. Please pick a slightly different time.")
        can_predict = False
        
elif mode == "Historical Date Lookup (Interactive)" and data is None:
    st.error("Dataset file 'dataset/delhi_pm25_aqi.csv' not found. Cannot look up historical dates.")
    can_predict = False
    
else:
    st.sidebar.header("Manual Historical Data")
    st.sidebar.markdown("Enter readings in **µg/m³**:")
    pm25_lag_3 = st.sidebar.number_input("PM2.5 (3 Hours Ago)", min_value=0.0, max_value=1200.0, value=150.0)
    pm25_lag_2 = st.sidebar.number_input("PM2.5 (2 Hours Ago)", min_value=0.0, max_value=1200.0, value=160.0)
    pm25_lag_1 = st.sidebar.number_input("PM2.5 (1 Hour Ago - Most Recent)", min_value=0.0, max_value=1200.0, value=170.0)


# Prediction Execution Action
if can_predict:
    if st.button(f"🔮 Forecast PM2.5 for {forecast_time_label}", type="primary"):
        payload = {
            "pm25_lag_1": pm25_lag_1,
            "pm25_lag_2": pm25_lag_2,
            "pm25_lag_3": pm25_lag_3
        }
        
        try:
            with st.spinner("Analyzing temporal patterns..."):
                response = requests.post(API_URL, json=payload)
                
            if response.status_code == 200:
                result = response.json()
                pm25_pred = result['prediction_pm25_next_hour']
                model_used = result.get('model_used', 'ML Model')
                
                st.success(f"### Predicted PM2.5 For {forecast_time_label}: **{pm25_pred:.2f} µg/m³**")
                st.markdown(f"*(Powered by **{model_used}**)*")
                
                # Indian AQI Scale
                if pm25_pred <= 30.0:
                    st.info("🟢 **Good** - Minimal Impact")
                elif pm25_pred <= 60.0:
                    st.info("🟡 **Satisfactory** - Minor breathing discomfort to sensitive people")
                elif pm25_pred <= 90.0:
                    st.warning("🟠 **Moderately Polluted** - Breathing discomfort to people with lungs, asthma and heart diseases")
                elif pm25_pred <= 150.0:
                    st.error("🔴 **Poor** - Breathing discomfort to most people on prolonged exposure")
                elif pm25_pred <= 250.0:
                    st.error("🟣 **Very Poor** - Respiratory illness on prolonged exposure")
                else:
                    st.error("🟫 **SEVERE** - Affects healthy people and seriously impacts those with existing diseases")
            else:
                st.error("Error from Backend API: " + str(response.json()))
        except Exception as e:
            st.error(f"Failed to connect to backend API: {e}")
            st.markdown("**Note:** Please Ensure the FastAPI Backend is running locally on port 8000.")
