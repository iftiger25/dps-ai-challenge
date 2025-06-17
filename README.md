# 🚨 AI-Powered Munich Traffic Accident Predictor 🚦  
*A machine learning model to forecast alcohol-related accidents in Munich (2021) using historical data (pre-2021).*  

![Historical Alcohol-Related Accidents (Munich) with 2021-01 Forecast](./historical_accidents_forecast.png)  

## 📌 Overview  
This project completes the **DPS AI Challenge** by:  
✅ **Cleaning and preprocessing** Munich’s traffic accident dataset (filtered to pre-2021 data).  
✅ **Training a Prophet time-series model** to predict accidents for *January 2021* (`Alkoholunfälle, insgesamt`).  
✅ **Deploying a FastAPI endpoint** hosted on Render.com.  
✅ **Computing error metrics** against actual 2021 data (optional validation).  

---

## 🚀 Features  
- **Data Pipeline**:  
  - Filters records post-2020 for consistency.  
  - Handles missing values and outliers.  
- **Forecasting Model**:  
  - Predicts alcohol-related accidents (`Alkoholunfälle, insgesamt`) for 2021.  
  - Supports custom year/month inputs via API.  
- **Live API**:  
  - Accepts JSON payloads (e.g., `{"year": 2021, "month": 1}`).  
  - Returns predictions (e.g., `{"prediction": 42}`).  

---

## ⚙️ Installation  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/iftiger25/ai-mvp-coach-challenge-munich-accidents.git
   cd ai-mvp-coach-challenge-munich-accidents
