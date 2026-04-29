# Bank Marketing Campaign — Classifier Comparison

Predicting term deposit subscriptions from a Portuguese bank's phone-based marketing data using four machine learning classifiers.

---

## Overview

A Portuguese bank conducted 17 phone-based marketing campaigns between May 2008 and November 2010, contacting over 41,000 clients to sell term deposit subscriptions. Only ~11% of calls resulted in a subscription.

This project builds and compares predictive models to identify which clients are most likely to subscribe, helping the bank focus its efforts and improve campaign ROI.

**Primary metric: ROC-AUC** — simple accuracy is misleading given the class imbalance (~89% / ~11%). ROC-AUC measures how well a model ranks likely subscribers above non-subscribers, which directly maps to prioritizing a call list.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

- 41,188 client contacts across 17 campaigns
- 20 input features (client info, campaign contact details, economic indicators)
- Binary target: did the client subscribe to a term deposit? (`yes` / `no`)
- `duration` (call length) is **excluded** from all models 

---

## Project Structure

```
├── MOD17_Bank_marketing_practical_app.ipynb   # Full analysis notebook
├── bank-additional-full.csv                   # Dataset (download from UCI)
└── README.md
```

---

## Analysis Sections

### 1. Business Understanding
Define the problem: predict which clients to call to maximize subscriptions while minimizing wasted outreach.

### 2. Data Loading & Exploration
Read in the dataset, inspect shape, dtypes, and class distribution.

### 3. Feature Understanding
Identify missing values (encoded as `"unknown"`), flag data leakage in `duration`, and note the `pdays` sentinel value of `999`.

### 4. Exploratory Data Analysis
- Subscription rates by job type, month, age, and prior campaign outcome
- Correlation heatmap of numeric features vs. target

### 5. Feature Engineering & Preprocessing
- Encode target: `yes → 1`, `no → 0`
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical features (unknown categories handled gracefully)
- `ColumnTransformer` + `Pipeline` for clean, leakage-free preprocessing

### 6. Train/Test Split
Stratified 80/20 split preserving the ~11% positive class rate in both sets.

### 7. Baseline
Majority-class baseline: **Accuracy ~0.89, ROC-AUC 0.50, F1 0.00** — the floor any useful model must beat.

### 8–9. Logistic Regression (Simple Model)
Build and evaluate a baseline logistic regression model. Interpret top coefficients.

### 10. Default Model Comparison
Compare all four classifiers at default settings across accuracy, ROC-AUC, F1, and training time.

| Model | Test ROC-AUC | Note |
|---|---|---|
| Logistic Regression | ~0.79 | Fast and interpretable |
| SVM (LinearSVC) | ~0.80 | Strong generalization |
| K-Nearest Neighbors | ~0.76 | Slower at scale |
| Decision Tree | ~0.74 | Overfits without tuning |

### 11. Hyperparameter Tuning
`GridSearchCV` with 5-fold stratified cross-validation improves all four models. Tuning is scored on ROC-AUC to align with the business objective.

---

## Key Findings

- **Prior campaign success** is the strongest predictor — clients who subscribed before convert at ~65% vs. ~8% for first-time contacts
- **Students and retirees** have the highest subscription rates by job category (~30–35%)
- **March, September, and October** yield 2–3x higher subscription rates than May (the highest-volume month)
- **Recommended model: Logistic Regression (tuned)** — achieves top ROC-AUC (~0.80), trains in seconds, and produces interpretable coefficients for business stakeholders

---
