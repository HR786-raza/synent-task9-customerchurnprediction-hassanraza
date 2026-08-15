# 📡 Customer Churn Prediction & Retention Analytics System

> **End-to-End Data Science Project** — Task 9 Internship Submission

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Business Problem

Customer churn — when a subscriber cancels their telecom service — is one of the most costly challenges in the industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a production-grade ML system to predict churn *before* it happens, enabling targeted retention campaigns that reduce revenue loss and improve customer lifetime value.

---

## 📋 Project Overview

A complete, production-style machine learning web application that:

- Predicts whether a telecom customer is likely to churn
- Provides a churn probability score (0–100%)
- Classifies risk level: Low / Medium / High
- Delivers actionable business retention recommendations
- Displays full model performance dashboard

---

## 📂 Dataset

**Telco Customer Churn Dataset** (IBM Sample Dataset / Kaggle)

| Property | Value |
|----------|-------|
| Rows | 7,043 customers |
| Columns | 21 features |
| Target | `Churn` (Yes / No) |
| Churn Rate | ~26.5% |

**Key Features:** Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

---

## 🏗️ Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.joblib       # Trained pipeline
│   └── metadata.json            # Metrics, feature importance, ROC data
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py         # Data loading & cleaning
│   ├── preprocessing.py         # Feature engineering & pipeline
│   └── train_model.py           # Training & model selection
├── app.py                       # Streamlit application
├── requirements.txt
└── README.md
```

---

## 🔬 Data Science Workflow

### 1. Data Cleaning
- Converted `TotalCharges` from string → numeric (11 rows had whitespace)
- Filled missing `TotalCharges` with `MonthlyCharges` for new customers (tenure = 0)
- Removed duplicate records
- Dropped `customerID` (non-predictive identifier)
- Encoded target: `Yes → 1`, `No → 0`

### 2. Exploratory Data Analysis
Visualisations produced:
- Churn distribution (pie chart)
- Churn by contract type
- Churn by tenure (histogram)
- Churn by monthly charges
- Churn by internet service
- Churn by payment method
- And more within the Streamlit dashboard

### 3. Feature Engineering
| Feature | Description |
|---------|-------------|
| `TenureGroup` | Categorical buckets: 0-12, 13-24, 25-48, 49-60, 61-72 months |
| `ServiceCount` | Number of add-on services subscribed |
| `HasStreaming` | Binary: subscribes to any streaming service |
| `AvgMonthlySpend` | TotalCharges / tenure (lifetime average) |

### 4. Preprocessing Pipeline
```
ColumnTransformer
├── Numerical → StandardScaler
└── Categorical → OneHotEncoder (handle_unknown='ignore')
```

Train/Test split: **80% / 20%** with stratification on `Churn`.

### 5. Models Trained & Compared

| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|---------|
| Logistic Regression | 0.738 | 0.786 | 0.842 |
| Decision Tree | 0.726 | 0.751 | 0.786 |
| Random Forest | 0.779 | 0.473 | 0.823 |
| Gradient Boosting | **0.804** | 0.503 | **0.844** |
| XGBoost | 0.754 | 0.757 | 0.838 |

**Selected model: Gradient Boosting** (highest ROC-AUC)

### 6. Evaluation — Business Meaning

| Metric | Value | Business Meaning |
|--------|-------|-----------------|
| Accuracy | 80.4% | 8 in 10 predictions correct |
| Precision | 67.6% | 2 in 3 flagged customers actually churn |
| Recall | 50.3% | Catches half of all churners |
| F1 Score | 57.7% | Balance between cost & coverage |
| ROC-AUC | **84.4%** | Strong discrimination ability |

> **Note:** A **False Negative** (missed churner) costs the business a full customer lifetime value. A **False Positive** (wrong flag) costs only one retention incentive. Recall is the priority metric.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Linux & macOS: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates churn_model.joblib & metadata.json)
python src/train_model.py

# 5. Launch the Streamlit app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## ☁️ Deployment — Streamlit Community Cloud

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and set **Main file path** to `app.py`
5. Click **Deploy**

Ensure `requirements.txt` is present and the `models/` folder (with `.joblib` and `.json`) is committed.

---

## 🖥️ Application Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI cards, EDA charts, key business insights |
| 🔮 Customer Prediction | Interactive form → churn probability + risk gauge + recommendations |
| 📊 Model Performance | Model comparison, confusion matrix, ROC curve, feature importance |
| ℹ️ About Project | Workflow, tech stack, dataset summary, project structure |

---

## 🔮 Future Improvements

- [ ] SHAP explainability values per prediction
- [ ] Batch CSV upload for bulk predictions
- [ ] Real-time CRM integration (Salesforce / HubSpot)
- [ ] Hyperparameter tuning with Optuna
- [ ] Model retraining pipeline with MLflow tracking
- [ ] Customer segmentation clustering
- [ ] Email alert system for high-risk customers

---

## 🛠️ Technologies

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Streamlit` · `Joblib`

---

## 👤 Author

Hassan Raza
AI & Data Science Intern

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
