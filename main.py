import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

file_path = 'monatszahlen_verkehrsunfaelle.csv'

print(f"Loading data from: {file_path}")
try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print("Data loaded successfully!")
    print("First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Make sure it's in the correct directory.")
    exit()

print("\nFiltering data to include records up to and including 2020...")
df_filtered = df[df['JAHR'] <= 2020].copy()
print(f"Original rows: {len(df)}, Rows after filtering: {len(df_filtered)}")

print("\nFiltering for 'Alkoholunfälle' and 'insgesamt'...")
df_specific = df_filtered[
    (df_filtered['MONATSZAHL'] == 'Alkoholunfälle') &
    (df_filtered['AUSPRAEGUNG'] == 'insgesamt')
].copy()

print(f"Rows for specific category/type: {len(df_specific)}")

df_specific = df_specific[df_specific['MONAT'] != 'Summe'].copy()
df_specific['MONAT_EXTRACTED'] = df_specific['MONAT'].astype(str).str[-2:]
df_specific['MONAT_NUM'] = pd.to_numeric(df_specific['MONAT_EXTRACTED'], errors='coerce')
df_specific.dropna(subset=['MONAT_NUM'], inplace=True)
df_specific['MONAT_NUM'] = df_specific['MONAT_NUM'].astype(int)
df_specific = df_specific[(df_specific['MONAT_NUM'] >= 1) & (df_specific['MONAT_NUM'] <= 12)].copy()

print(f"Rows after cleaning and filtering 'MONAT' column: {len(df_specific)}")

df_specific['MONAT_PADDED'] = df_specific['MONAT_NUM'].astype(str).str.zfill(2)
df_specific['Date'] = pd.to_datetime(
    df_specific['JAHR'].astype(str) + '-' + df_specific['MONAT_PADDED'] + '-01',
    format='%Y-%m-%d'
)
df_specific.set_index('Date', inplace=True)
df_specific.sort_index(inplace=True)

df_model_data = df_specific[['WERT']].copy()

df_prophet_format = df_model_data.reset_index().rename(columns={'Date': 'ds', 'WERT': 'y'})
df_prophet_format['ds'] = pd.to_datetime(df_prophet_format['ds'])
df_prophet_format['y'] = pd.to_numeric(df_prophet_format['y'], errors='coerce')
df_prophet_format.dropna(subset=['y'], inplace=True)

print("\nInitializing and training Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)
model.fit(df_prophet_format)
print("Model training complete.")

print("\nMaking prediction for January 2021...")
future = model.make_future_dataframe(periods=1, freq='MS', include_history=False)
forecast = model.predict(future)
predicted_value_2021_01 = forecast['yhat'].iloc[0]
print(f"Predicted number of 'Alkoholunfälle' for 2021-01 (insgesamt): {predicted_value_2021_01:.2f}")

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_prophet_format, x='ds', y='y', marker='o', label='Historical Accidents (Up to 2020)')
plt.plot(forecast['ds'].iloc[0], predicted_value_2021_01, marker='X', color='red', markersize=10, label='Predicted 2021-01')

plt.title('Historical Alcohol-Related Accidents (Munich) with 2021-01 Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Accidents (WERT)')
plt.grid(True)
plt.legend()
plt.tight_layout()

image_filename = 'historical_accidents_forecast.png'
plt.savefig(image_filename)
print(f"Visualization saved as '{image_filename}'")
plt.show()

model_filename = 'prophet_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained Prophet model saved as '{model_filename}'")