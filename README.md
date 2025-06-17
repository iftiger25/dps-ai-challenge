# 🚨 AI-Powered Munich Traffic Accident Predictor 🚦  
*A machine learning model to forecast alcohol-related accidents in Munich (2021) using historical data (pre-2021).*  

![Historical Alcohol-Related Accidents (Munich) with 2021-01 Forecast](./historical_accidents_forecast.png)  

## 📌 Overview  
This project completes the **DPS AI Challenge** by:  
✅ **Cleaning and preprocessing** Munich’s traffic accident dataset (filtered to pre-2021 data).  
✅ **Training a Prophet time-series model** to predict accidents for *January 2021* (`Alkoholunfälle, insgesamt`).  
✅ **Deploying a FastAPI endpoint** hosted on Render.com.  
✅ **Computing error metrics** against actual 2021 data (optional validation).  


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

2. **Install dependencies:**
```bash
pip install -r requirements.txt

##🏃 Usage
1. **Training the Model**
```bash
python train.py --data_path ./data/verkehrsunfaelle.csv
(Outputs model weights and evaluation metrics.)

2. **Running the API Locally**
```bash
uvicorn src.api:app --reload
Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2021, "month": 1}'
Response:

json
{"prediction": 35}

3. **Live API Documentation**
🔗 Deployed API: https://dps-ai-challenge-n0s4.onrender.com/docs

Endpoint: /predict
Method: POST
Input: {"year": 2021, "month": 1}

# 📂 Project Structure

├── data/
│   └── verkehrsunfaelle.csv          # Raw dataset (filtered to pre-2021)
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory analysis
│   └── Model_Training.ipynb          # Prophet model experiments
├── src/
│   ├── train.py                      # Training script
│   └── api.py                        # FastAPI endpoint
├── outputs/
│   ├── model.pkl                     # Saved Prophet model
│   └── historical_accidents_forecast.png  # Prediction visualization
├── requirements.txt
└── README.md

# 🌐 Deployment
Tech Stack:
• Framework: FastAPI
• Model: Prophet (Facebook’s time-series library)
• Hosting: Render.com

# 🤝 Contributing
1. Fork the repository.
1. Open an issue or submit a PR with improvements.

# 📜 License
MIT
