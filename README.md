# ✈️ Flight Operations Intelligence

An end-to-end aviation analytics and machine learning project for exploring flight delays, airline performance, route reliability, delay causes, airport operational risk, and predicted delay probability.

The project combines:

* Python
* Pandas
* SQL
* Scikit-learn
* Streamlit
* Matplotlib
* Machine Learning
* Operational Analytics

The goal is to transform raw flight data into meaningful operational insights and provide an interactive dashboard for aviation delay analysis.

## 🚀 Project Overview

Airlines and airports generate large amounts of operational data every day.

This project analyzes flight operations data to answer questions such as:

* Which airlines experience the highest delay rates?
* Which airports have the highest operational delay risk?
* Which routes are frequently delayed?
* What are the major causes of flight delays?
* How do delay patterns change over time?
* Can machine learning estimate whether a flight is likely to arrive late?

The system has been designed as a modular aviation analytics and delay-risk intelligence platform rather than a single analysis notebook.

## 🎯 Main Features

### Operational KPI Analysis

The dashboard calculates:

* Total flights
* Delayed flights
* Delay rate
* Severe delay rate
* Average arrival delay
* Median arrival delay
* Maximum arrival delay

A flight is considered delayed when:

```text
Arrival Delay >= 15 minutes
```

A severe delay is defined as:

```text
Arrival Delay >= 60 minutes
```

## ✈️ Airline Performance Analysis

Airlines are analyzed using:

* Total flights
* Delayed flights
* Delay rate
* Severe delays
* Severe delay rate
* Average arrival delay
* Median arrival delay
* Maximum delay

This provides a clearer view of airline reliability instead of relying only on average delay.

## 🗺️ Route Intelligence

The project evaluates routes using:

* Number of flights
* Number of delayed flights
* Delay rate
* Average arrival delay
* Maximum delay

Minimum flight thresholds are used to reduce misleading rankings caused by very small sample sizes.

## ⚠️ Delay Cause Analysis

Delay minutes are categorized into:

* Carrier delay
* Weather delay
* National Airspace System delay
* Security delay
* Late aircraft delay

The system also calculates the percentage contribution of each delay category.

Example:

```text
Late Aircraft      → 32%
NAS                → 30%
Carrier            → 28%
Weather            → 10%
```

This helps identify the dominant operational causes of flight disruption.

## 🛫 Airport Operational Risk Score

A custom Airport Operational Risk Score is used to compare airports.

The score considers:

* Delay rate
* Severe delay rate
* Average departure delay

Because these metrics use different scales, each metric is first normalized to a range of:

```text
0 – 100
```

The final score uses:

```text
50% Delay Rate
30% Severe Delay Rate
20% Average Departure Delay
```

Airports are categorized as:

```text
0 – 25       LOW
25 – 50      MEDIUM
50 – 75      HIGH
75 – 100     CRITICAL
```

Important:

> The Airport Operational Risk Score measures relative delay-related operational risk within the dataset. It is not an aviation safety rating.

## 🤖 Machine Learning Delay Prediction

The project includes an ML-based flight delay predictor.

The model estimates the probability that a flight will arrive at least 15 minutes late.

### Input Features

Only information available before departure is used.

Categorical features include:

```text
Airline
Origin Airport
Destination Airport
Month
Day of Week
Time of Day
```

Numerical features include:

```text
Departure Hour
Departure Minute
Distance
Weekend Indicator
```

Features such as these are intentionally excluded:

```text
ARR_DELAY
DEP_DELAY
DELAY_DUE_CARRIER
DELAY_DUE_WEATHER
DELAY_DUE_NAS
```

because including them would create data leakage.

## 🧠 Feature Engineering

Several useful features are created from the original dataset.

### Date Features

```text
YEAR
MONTH
MONTH_NAME
DAY_OF_WEEK
IS_WEEKEND
```

### Route Feature

```text
ORIGIN → DEST
```

Example:

```text
JFK → LAX
```

### Departure Time Features

Instead of using raw scheduled time such as:

```text
1835
```

the system converts it into:

```text
DEP_HOUR = 18
DEP_MINUTE = 35
TIME_OF_DAY = Evening
```

Time-of-day categories include:

```text
Morning
Afternoon
Evening
Night
```

## 🧪 Machine Learning Models Compared

Three classification models are evaluated:

1. Logistic Regression
2. Random Forest
3. Hist Gradient Boosting

The models are compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

A custom model-selection score is also used:

```text
40% ROC-AUC
30% Recall
20% F1 Score
10% Precision
```

Recall is given additional importance because detecting delayed flights is more useful than simply maximizing overall accuracy.

## 📊 Current Model Results

Using the current 1,000-flight sample dataset:

### Logistic Regression

```text
Accuracy:   0.690
Precision:  0.308
Recall:     0.541
F1 Score:   0.392
ROC-AUC:    0.674
```

### Random Forest

```text
Accuracy:   0.740
Precision:  0.297
Recall:     0.297
F1 Score:   0.297
ROC-AUC:    0.680
```

### Hist Gradient Boosting

```text
Accuracy:   0.780
Precision:  0.182
Recall:     0.054
F1 Score:   0.083
ROC-AUC:    0.579
```

### Selected Model

The current best model is:

```text
Logistic Regression
```

Although some models achieve higher raw accuracy, Logistic Regression performs better at identifying actual delayed flights.

Current delayed-flight recall:

```text
54.1%
```

## 📉 Why Accuracy Alone Is Misleading

The dataset is imbalanced.

Example:

```text
Not Delayed: 814
Delayed:     186
```

A model that predicts every flight as "not delayed" could still achieve high accuracy.

Therefore, the project prioritizes:

```text
Recall
F1 Score
ROC-AUC
```

along with accuracy.

## 📊 Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Dashboard sections include:

### Operational Overview

Displays:

* Total flights
* Delayed flights
* Delay rate
* Severe delay rate
* Average delay
* Median delay
* Maximum delay

### Key Operational Insights

Automatically identifies:

* Airline with the highest delay rate
* Dominant delay cause
* Airport with the highest operational risk

### Airline Analysis

Interactive airline reliability comparison.

### Airport Risk Analysis

Displays the highest operational-risk airports.

### Route Intelligence

Ranks frequently delayed routes.

### Delay Cause Analysis

Shows total delay minutes and percentage contribution.

### Monthly Trends

Displays delay-rate trends across months.

### ML Delay Predictor

Users can enter:

```text
Airline
Origin
Destination
Month
Day
Scheduled departure time
Distance
```

and receive:

```text
Predicted Delay Probability

Risk Level:
LOW / MEDIUM / HIGH
```

## 🏗️ Project Architecture

```text
flight-operations-intelligence/
│
├── data/
│   └── sample_flight_data.csv
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── generated model files
│
├── sql/
│   └── flight_analysis.sql
│
├── src/
│   ├── data_cleaning.py
│   ├── analysis.py
│   ├── visualization.py
│   └── train_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔄 Data Pipeline

```text
Raw Flight Dataset
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Operational KPI Analysis
        ↓
Airline / Route / Airport Analysis
        ↓
Operational Risk Scoring
        ↓
Machine Learning Training
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Interactive Streamlit Dashboard
```

## 🧹 Data Cleaning

The preprocessing pipeline handles:

* Date conversion
* Duplicate detection
* Missing delay values
* Delay feature creation
* Route generation
* Time feature extraction

Flight dates are converted from:

```text
DD-MM-YYYY
```

into Pandas datetime format.

Missing values in delay-reason columns are treated as zero where no delay reason was recorded.

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/flight-operations-intelligence.git
```

Move into the project:

```bash
cd flight-operations-intelligence
```

Create a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 📦 Requirements

Example:

```text
pandas
matplotlib
streamlit
scikit-learn
joblib
```

## 🧪 Run the Data Analysis

```bash
python src/analysis.py
```

This displays:

* Operational KPIs
* Airline performance
* Delayed routes
* Delay cause analysis
* Airport operational risk

## 📊 Run Visualizations

```bash
python src/visualization.py
```

This generates charts for:

* Airline delay rates
* Delay causes
* Delayed routes
* Airport risk

## 🤖 Train the Machine Learning Model

```bash
python src/train_model.py
```

Training flow:

```text
Load data
↓
Create features
↓
Split train/test data
↓
Train multiple models
↓
Evaluate models
↓
Compare performance
↓
Select the best model
↓
Save the best model
```

Generated model files are stored inside:

```text
models/
```

## 🌐 Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The application should open locally, usually at:

```text
http://localhost:8501
```

## 🔎 SQL Analysis

The project also contains SQL queries for:

* Total flights
* Flights by airline
* Average delays
* Airport delay analysis
* Route analysis
* Cancellation analysis

SQL file:

```text
sql/flight_analysis.sql
```

## 📁 Dataset

The repository contains a smaller sample dataset for demonstration:

```text
data/sample_flight_data.csv
```

Current sample:

```text
1,000 flights
34 original columns
```

The sample is sufficient for demonstrating the analytics pipeline and dashboard but is relatively small for advanced machine-learning conclusions.

## ⚠️ Current Limitations

### Small Dataset

The current ML model is trained on only 1,000 sample flights.

This limits its ability to learn complex:

* Route patterns
* Airport patterns
* Seasonal effects
* Airline-specific behavior

### Class Imbalance

Delayed flights represent a minority of the dataset.

This makes delayed-flight prediction more difficult.

### Relative Risk Score

The airport risk score is relative to the airports contained in the current dataset.

It should not be interpreted as an official airport risk or safety metric.

### ML Predictions

Predictions are experimental and intended for:

```text
Analytics demonstration
Machine learning experimentation
Portfolio use
Educational purposes
```

They should not be used for real-world airline operational decisions.

## 🔮 Future Improvements

Planned improvements include:

* Training on a much larger flight dataset
* Better class-imbalance handling
* Cross-validation
* Hyperparameter tuning
* Probability threshold optimization
* Feature importance analysis
* SHAP explainability
* Historical airline delay features
* Historical airport delay features
* Historical route reliability features
* Geographic airport maps
* Real-time weather integration
* Airport congestion features
* Model monitoring
* Cloud deployment
* PostgreSQL integration
* Automated data pipeline

## 💡 Potential Advanced Architecture

```text
Weather Data
      ↓
Airport Congestion
      ↓
Historical Route Reliability
      ↓
Airline Performance
      ↓
Scheduled Departure Information
      ↓
Machine Learning Model
      ↓
Flight Delay Probability
```

## 🧰 Tech Stack

### Data Analysis

```text
Python
Pandas
```

### Machine Learning

```text
Scikit-learn
Joblib
```

### Visualization

```text
Matplotlib
Streamlit
```

### Database Analysis

```text
SQL
```

### Development

```text
VS Code
Git
GitHub
```

## 📚 Skills Demonstrated

This project demonstrates practical experience with:

* Data cleaning
* Exploratory data analysis
* Feature engineering
* Pandas
* SQL analytics
* Operational KPI design
* Data visualization
* Classification models
* Class imbalance
* Model evaluation
* ROC-AUC
* Precision / Recall / F1
* ML pipelines
* One-hot encoding
* Model persistence
* Streamlit
* Modular Python architecture
* Git and GitHub

## 👨‍💻 Project Focus

This project focuses on aviation operations analytics, delay intelligence, operational risk analysis, and predictive machine learning.

## ⭐ Project Status

```text
Data Pipeline             ✅
Feature Engineering       ✅
Airline Analytics         ✅
Route Analytics           ✅
Delay Cause Analysis      ✅
Airport Risk Scoring      ✅
Streamlit Dashboard       ✅
ML Model Comparison       ✅
Delay Prediction          ✅
Large Dataset Training    🔄
Deployment                🔄
```

If you find this project useful, consider giving the repository a ⭐.

---
### Abhishek Rawat ❤️