# Predicting Arrests in Chicago Crime Data

This project builds supervised machine learning models to predict whether a reported crime incident in Chicago will result in an arrest (`Arrest = True/False`). The goal is to identify key factors associated with arrest likelihood and compare different modeling approaches.

**Course/Context:** MS in Artificial Intelligence & Business Analytics, University of South Florida — Machine Learning (Nov 2024)  
**Domain:** Public Safety & Law Enforcement

---

## Problem Statement
Using historical Chicago crime records, predict the probability of an arrest for a given crime incident to support:
- smarter resource allocation,
- proactive interventions in high-risk areas/times,
- operational efficiency through automated analysis.

---

## Data Sources
- **Chicago Data Portal — Crimes 2001 to Present**
- **Community Areas dataset** (used to map community area numbers to community names)

> Dataset includes crime type, location description, domestic flag, community area, year/time fields, and arrest outcome.

---

## Dataset Overview
- ~220K+ rows and ~22 features (crime incidents + attributes)
- Target variable: **`Arrest`** (boolean)

**Key input features used**
- `Primary Type` (crime category)
- `Location Description`
- `Community Area`
- `Year`
- `Domestic`

---

## Approach
### 1) Exploratory Data Analysis (EDA)
- Identified most common crime types and community-level patterns.
- Observed relationships between domestic-related crimes and arrest outcomes.

### 2) Preprocessing
- Removed minimal missing values (<1% in some location/coordinate fields)
- Addressed class imbalance (approx. **80% non-arrest vs 20% arrest**) via **oversampling**
- One-hot encoded categorical variables (e.g., `Primary Type`, `Location Description`)
- Scaled numeric features using Min-Max scaling
- Train/Test split: **80/20** (stratified)

---

## Models Evaluated
- **Logistic Regression** (interpretable baseline)
- **Random Forest** (non-linear ensemble + feature importance)
- **Neural Network (MLP Classifier)** (handles complex relationships in high-dimensional feature space)

---

## Results (Test Set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.75 | 0.74 | 0.76 | 0.75 | 0.84 |
| Random Forest | 0.74 | 0.72 | 0.78 | 0.75 | 0.83 |
| Neural Network (MLP) | **0.79** | **0.80** | 0.76 | **0.78** | **0.88** |

**Best model:** Neural Network (MLP), based on strongest overall performance (accuracy + ROC-AUC + balanced metrics).

---

## How to Run
### Option A: Notebook
1. Open `notebooks/Final_ML_Project.ipynb`
2. Install dependencies (see below)
3. Run cells end-to-end

### Dependencies
Create and activate a virtual environment (recommended), then:
```bash
pip install -r requirements.txt
