# Chicago Crime Arrest Prediction

This project applies supervised machine learning techniques to predict whether a reported crime incident in Chicago will result in an arrest (`Arrest = True / False`). The objective is to identify key factors associated with arrest likelihood and to compare multiple classification models using historical crime data.

**Course Context:**  
MS in Artificial Intelligence & Business Analytics – Machine Learning  
University of South Florida

---

## Problem Statement
Law enforcement agencies often operate with limited resources while managing large volumes of crime data. Understanding patterns that influence arrest outcomes can help support data-driven analysis, operational insights, and more efficient prioritization of cases.

This project aims to predict arrest outcomes based on crime characteristics such as crime type, location, time, and community area.

---

## Data Sources
- **Chicago Data Portal – Crimes 2001 to Present**
- **Community Areas Dataset** (used for mapping community area identifiers)

---

## Dataset Overview
- ~220,000+ crime records
- ~22 features after preprocessing
- Target variable: **`Arrest`** (binary classification)

---

## Approach

### Exploratory Data Analysis (EDA)
- Analyzed crime distribution by type, location, and community area
- Observed correlations between domestic crimes and arrest likelihood

### Data Preprocessing
- Removed minimal missing values
- Addressed class imbalance (~80% non-arrest vs 20% arrest) using oversampling
- One-hot encoded categorical variables
- Applied Min-Max scaling
- Performed an 80/20 train-test split

---

## Models Evaluated
- **Logistic Regression**
- **Random Forest**
- **Neural Network (MLP Classifier)**

---

## Results (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.75 | 0.74 | 0.76 | 0.75 | 0.84 |
| Random Forest | 0.74 | 0.72 | 0.78 | 0.75 | 0.83 |
| Neural Network (MLP) | **0.79** | **0.80** | 0.76 | **0.78** | **0.88** |

---

## How to Run

- Clone or download this repository.
- Open the Jupyter notebook: `notebooks/Final_ML_Project.ipynb`
- Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn




Repository Contents

notebooks/Final_ML_Project.ipynb — Full workflow (EDA → preprocessing → modeling → evaluation)

docs/ML_Documentation.pdf — Detailed project documentation

docs/Chicago_Crime_Project_Deck.pdf — Presentation deck

Notes / Ethical Considerations

This project uses historical arrest data, which may reflect underlying reporting or enforcement biases. Any real-world use should include fairness checks, bias audits, and stakeholder review.

Team & Contributions

Team project completed as part of a USF Machine Learning course. Contributions included data preparation, model development, documentation, and presentation.

License

This project is licensed under the MIT License.
