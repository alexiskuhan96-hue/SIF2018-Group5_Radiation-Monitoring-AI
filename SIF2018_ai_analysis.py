import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------
# 1. DATA COLLECTION & PREPROCESSING
# ---------------------------------------------------------
# Load the dataset
df = pd.read_csv('/content/SIF2018_radiation_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Data Cleaning: Handle the 11:00 AM Missing Data (Sensor Fault)
# We use forward-fill to temporarily substitute the missing reading 
# with the last known good reading so the AI doesn't crash.
df['gamma_dose_rate'] = df['gamma_dose_rate'].ffill()

# ---------------------------------------------------------
# 2. AI INTEGRATION (THE DECISION ENGINE)
# ---------------------------------------------------------
# We feed the AI three features: Radiation, Temperature, and Rainfall.
# This ensures the AI knows that a slight spike during a rainstorm is natural.
features = df[['gamma_dose_rate', 'temperature', 'rainfall']]

# Initialize the Isolation Forest AI Model
# It isolates "anomalies" that differ vastly from normal patterns.
ai_model = IsolationForest(contamination=0.1, random_state=42)

# The AI predicts if a row is normal (1) or an anomaly (-1)
df['ai_anomaly_score'] = ai_model.fit_predict(features)

# ---------------------------------------------------------
# 3. ALERT WORKFLOW & REPORTING
# ---------------------------------------------------------
def determine_alert_status(row):
    # If the AI flags an anomaly AND radiation is dangerously high
    if row['ai_anomaly_score'] == -1 and row['gamma_dose_rate'] > 150:
        return '🔴 ALERT: Notify Authority'
    # If radiation is climbing but not quite an emergency
    elif row['gamma_dose_rate'] > 90:
        return '🟡 WATCH: Monitor Closely'
    # Everything is fine
    else:
        return '🟢 NORMAL'

# Apply our logic to the dataset
df['system_status'] = df.apply(determine_alert_status, axis=1)

# Display the final analyzed report specifically for the Perimeter Station (P1)
p1_report = df[df['station_id'] == 'P1'][['timestamp', 'gamma_dose_rate', 'system_status']]
print(p1_report.to_string(index=False))