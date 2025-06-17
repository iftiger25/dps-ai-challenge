# 🚨 AI-Powered Munich Traffic Accident Predictor 🚦  
*A machine learning model to forecast alcohol-related accidents in Munich (2021) using historical data (pre-2021).*  

![Demo GIF/Screenshot](#) *(Add a screenshot of your visualization/prediction here)*  

## 📌 Overview  
This project completes the **DPS AI Challenge** by:  
✅ **Cleaning and preprocessing** Munich’s traffic accident dataset (filtered to pre-2021 data).  
✅ **Training a time-series model** to predict accidents for *January 2021* (`Alkoholunfälle, insgesamt`).  
✅ **Deploying an API endpoint** that returns predictions via POST requests.  
✅ **Computing error metrics** against actual 2021 data (optional validation).  

---

## 🚀 Features  
- **Data Pipeline**:  
  - Filters records post-2020 for consistency.  
  - Handles missing values and outliers.  
- **Forecasting Model**:  
  - Predicts alcohol-related accidents (`Alkoholunfälle, insgesamt`) for 2021.  
  - Supports custom year/month inputs via API.  
- **API Deployment**:  
  - Accepts JSON payloads (e.g., `{"year": 2021, "month": 1}`).  
  - Returns predictions (e.g., `{"prediction": 42}`).  

---

## ⚙️ Installation  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/iftiger25/ai-mvp-coach-challenge-munich-accidents.git
   cd ai-mvp-coach-challenge-munich-accidents

2. **Install dependencies**:
   pip install -r requirements.txt
   
3. **Set up environment variables**
echo "OPENAI_API_KEY=your_key" > .env  # If using LLMs

## 🏃 Usage
python train.py --data_path ./data/verkehrsunfaelle.csv
1. Training the Model
   python train.py --data_path ./data/verkehrsunfaelle.csv
2. Running the API Locally
