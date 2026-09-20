"# # AI-Based Demand Forecasting System for Retail Sales

## 1. Project Overview

This project implements an end-to-end AI/ML demand forecasting system for retail sales.

The objective is to forecast future product demand using historical sales data, product information, store information, promotions, calendar features, lag features, and rolling statistics.

The system includes:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Time-series data splitting
* Seasonal baseline forecasting
* XGBoost machine learning model
* Model evaluation
* Recursive multi-day forecasting
* Streamlit dashboard
* Automated tests

---

## 2. Dataset

The project uses the Kaggle Store Sales - Time Series Forecasting dataset.

The dataset contains retail sales information for multiple stores and product families.

Important columns include:

* `date` — sales date
* `store_nbr` — store identifier
* `family` — product family
* `sales` — sales quantity
* `onpromotion` — number of products on promotion

The raw training dataset contains approximately 3 million records covering multiple stores and product families.

---

## 3. System Architecture

```text
Sales Data
    |
    v
Data Validation & Cleaning
    |
    v
Feature Engineering
    |
    +--> Calendar Features
    +--> Lag Features
    +--> Rolling Features
    |
    v
Chronological Data Split
    |
    +--> Training Data
    +--> Validation Data
    +--> Test Data
    |
    +-------------------+
    |                   |
    v                   v
7-Day Baseline       XGBoost Model
    |                   |
    +---------+---------+
              |
              v
        Model Evaluation
              |
              v
       Trained Forecast Model
              |
              v
      Recursive Forecasting
              |
              v
       Streamlit Dashboard
```

---

## 4. Feature Engineering

The model uses three major groups of features.

### Calendar Features

* Day of week
* Day of month
* Month
* Week of year
* Year

### Lag Features

* Lag 1 day
* Lag 7 days
* Lag 14 days

Lag features allow the model to learn from previous demand.

### Rolling Features

* 7-day rolling mean
* 14-day rolling mean
* 30-day rolling mean

Rolling features summarize recent demand patterns.

The rolling calculations are based on shifted historical values to avoid target leakage.

---

## 5. Data Splitting

Because this is a time-series forecasting problem, the dataset is split chronologically rather than randomly.

### Training

Through:

```text
2016-12-31
```

### Validation

```text
2017-01-01 to 2017-06-30
```

### Test

```text
2017-07-01 to 2017-08-15
```

This prevents future observations from being used to train the model.

---

## 6. Baseline Model

A 7-day seasonal baseline is used.

The baseline predicts demand using the sales value from seven days earlier:

```text
prediction = lag_7
```

Test performance:

| Metric | Baseline |
| ------ | -------: |
| MAE    |    88.63 |
| RMSE   |   331.07 |
| MAPE   |   46.72% |

The baseline provides a simple reference point for evaluating the machine learning model.

---

## 7. XGBoost Model

The main forecasting model is an XGBoost regression model.

The model uses:

* Store number
* Product family
* Promotion information
* Calendar features
* Lag features
* Rolling demand features

Model configuration:

```text
n_estimators = 300
max_depth = 8
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
tree_method = hist
```

---

## 8. Model Performance

The final model was evaluated on the unseen test period.

| Model          |   MAE |   RMSE |   MAPE |
| -------------- | ----: | -----: | -----: |
| 7-Day Baseline | 88.63 | 331.07 | 46.72% |
| XGBoost        | 58.86 | 217.12 | 42.71% |

Compared with the baseline on this test period:

* MAE improved by **33.59%**
* RMSE improved by **34.42%**
* MAPE improved by **8.58%**

These results are specific to the selected dataset and test period.

---

## 9. Important Features

The most influential features in the trained model include:

1. `rolling_mean_7`
2. `lag_7`
3. `lag_14`
4. `lag_1`
5. `day_of_week`
6. `rolling_mean_30`

The feature importance results indicate that recent historical demand is particularly important for forecasting future sales.

---

## 10. Recursive Forecasting

The system supports multi-day forecasting.

For example:

```text
7-day forecast
14-day forecast
30-day forecast
```

For future dates where actual sales are not available, the system recursively uses previous predictions to construct future lag and rolling features.

This allows the model to generate a complete future demand trajectory.

---

## 11. Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows users to select:

* Store
* Product family
* Forecast start date
* Forecast horizon
* Promotion level

The dashboard displays:

* Daily demand forecast
* Total predicted demand
* Average predicted demand
* Maximum predicted demand
* Minimum predicted demand
* Forecast chart
* Forecast table
* CSV download

---

## 12. Project Structure

```text
demand-forecasting-ai/
│
├── data/
│   ├── train.csv
│   ├── features.pkl
│   ├── train_features.pkl
│   ├── validation_features.pkl
│   ├── test_features.pkl
│   ├── baseline_results.csv
│   ├── xgboost_validation_results.csv
│   ├── test_predictions.pkl
│   ├── final_model_comparison.csv
│   └── visualizations
│
├── notebooks/
│   └── demand_forecasting.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── split_data.py
│   ├── baseline.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── forecast.py
│   └── visualize.py
│
├── models/
│   ├── demand_model.pkl
│   └── family_mapping.pkl
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   └── test_forecasting.py
│
├── system_design/
│   └── architecture.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 13. Installation

Clone or download the project and open a terminal in the project directory.

Create a virtual environment:

```bash
py -3.13 -m venv .venv
```

Activate it on Windows Command Prompt:

```bash
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 14. Running the Pipeline

Feature engineering:

```bash
python src/features.py
```

Split the data:

```bash
python src/split_data.py
```

Run the baseline:

```bash
python src/baseline.py
```

Train XGBoost:

```bash
python src/train.py
```

Evaluate the models:

```bash
python src/evaluate.py
```

Generate visualizations:

```bash
python src/visualize.py
```

Run tests:

```bash
pytest -q
```

---

## 15. Running the Dashboard

Start Streamlit:

```bash
streamlit run app\streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

---

## 16. Testing

The forecasting system includes automated tests.

Current test result:

```text
7 passed
```

The tests verify important forecasting functionality and help ensure that changes do not break the forecasting pipeline.

---

## 17. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Plotly
* Joblib
* Streamlit
* Pytest

---

## 18. Conclusion

This project demonstrates a complete retail demand forecasting pipeline from raw historical sales data to an interactive forecasting application.

The system combines time-series feature engineering with gradient-boosted machine learning and provides both model evaluation and practical multi-day forecasting functionality.

The XGBoost model achieved lower MAE and RMSE than the 7-day seasonal baseline on the selected unseen test period.
" 
