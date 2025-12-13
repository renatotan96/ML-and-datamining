# 🤖 Machine Learning & Data Mining Portfolio

> A comprehensive collection of machine learning projects demonstrating classification, regression, clustering, and association rule mining with complete end-to-end workflows.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Technologies](#-technologies)
- [Projects](#-projects)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
- [Key Achievements](#-key-achievements)
- [Quick Start](#-quick-start)

---

## 🎯 Overview

This repository showcases practical machine learning applications across multiple domains including healthcare, meteorology, retail, and customer analytics. Each project demonstrates complete ML workflows:

- **Exploratory Data Analysis (EDA)**
- **Data Preprocessing & Feature Engineering**
- **Model Training & Comparison**
- **Evaluation & Interpretation**

---

## 🛠️ Technologies

**Language:** Python 3.8+

**Libraries:**
```
scikit-learn  │  pandas  │  numpy  │  matplotlib  │  seaborn  │  SMOTE  │  mlxtend
```

---

## 📊 Projects

### Supervised Learning

<details open>
<summary><b>🏥 Fetal Health Classification</b></summary>

**Objective:** Predict fetal health status from Cardiotocogram (CTG) data to reduce child and maternal mortality

**Approach:**
- Performed correlation analysis to identify key predictive features
- Compared 4 classification algorithms

**Models Tested:**
- ✅ **Random Forest** (Best Performance)
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

**Key Features:**
- Accelerations
- Prolonged decelerations
- Abnormal short-term variability
- Percentage of long-term variability

**Results:** Random Forest achieved highest accuracy with clear pattern identification

</details>

<details open>
<summary><b>🌧️ Rainfall Prediction (Australia)</b></summary>

**Objective:** Predict next-day rainfall using 10 years of weather observations

**Preprocessing:**
- Handled missing data strategically
- Applied label encoding for categorical variables
- Used SMOTE to address class imbalance

**Models Tested:**
- ✅ **Random Forest** (Best: 84% accuracy, 64% precision)
- Gradient Boosting
- Logistic Regression
- K-Nearest Neighbors

**Top Predictive Features:**
- Humidity at 3pm
- Humidity at 9am
- Rain today (binary)
- Rainfall amount

</details>

---

### Regression

<details>
<summary><b>🍦 Ice Cream Revenue Prediction</b></summary>

**Objective:** Predict daily revenue based on temperature

**Method:** Simple Linear Regression

**Results:**
- **R² Score:** 97.73%
- **Mean Absolute Error:** $19.08
- Demonstrated strong linear relationship between temperature and revenue

</details>

<details>
<summary><b>🚗 Car Price Prediction</b></summary>

**Objective:** Predict vehicle prices based on specifications

**Approach:**
- Converted categorical features to numerical values
- Performed correlation analysis
- Applied Multiple Linear Regression

**Key Price Determinants:**
- Horsepower
- Engine Size
- Curb Weight
- Car Width & Length

**Results:**
- **Model Accuracy:** 79.83%
- Successfully identified primary pricing factors

**Future Improvements:**
- Apply R² score for variance analysis
- Revisit feature selection for improved MAE
- Explore additional feature combinations

</details>

---

### Unsupervised Learning

<details open>
<summary><b>👥 Customer Segmentation Analysis</b></summary>

**Objective:** Identify customer segments for targeted marketing strategies

**Methods:**
- Applied Elbow Method to determine optimal cluster count
- Implemented K-Means Clustering
- Validated with DBSCAN

**Results:**
- **5 Distinct Segments Identified:**
  - High Income, High Spending
  - High Income, Low Spending
  - Mid Income, Mid Spending
  - Low Income, High Spending
  - Low Income, Low Spending

**Business Impact:** Provided actionable insights for marketing team targeting

</details>

<details>
<summary><b>🛒 Market Basket Analysis - Online Retail</b></summary>

**Objective:** Discover product associations for cross-selling strategies

**Approach:**
- Focused analysis on UK market
- Removed cancelled orders from dataset
- Applied Apriori algorithm for association rules

**Results:**
- Successfully identified frequent itemsets
- Discovered high-confidence product associations
- Generated recommendations for product bundling and store layout optimization

</details>

<details>
<summary><b>🚢 Titanic Survival Analysis</b></summary>

**Objective:** Analyze demographic patterns in survival rates

**Method:** Association rule mining on passenger data

**Key Findings:**
- ♀️ Higher female survival rates across all passenger classes
- 🎫 Passenger class significantly influenced survival probability
- 📊 Clear demographic patterns in survival outcomes

</details>

---

## 🏆 Key Achievements

| Metric | Achievement |
|--------|-------------|
| **Model Accuracy** | Consistently >79% across classification tasks |
| **Data Handling** | Effective management of missing values and class imbalance |
| **Algorithm Diversity** | Applied 10+ ML algorithms across various domains |
| **Business Value** | Delivered actionable insights for healthcare, retail, and marketing |

---

## 💡 Core Competencies
```
Data Cleaning & Preprocessing  •  Exploratory Data Analysis (EDA)
Feature Engineering & Selection  •  Classification & Regression Modeling
Clustering Analysis  •  Association Rule Mining
Model Evaluation & Validation  •  Data Visualization
Business Insight Generation
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ml-data-mining-portfolio.git
cd ml-data-mining-portfolio

# Install required packages
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Dataset Setup

Datasets are not included due to size constraints. Download from original sources listed in each notebook and place in the `data/` folder.

---

## 📁 Repository Structure
```
ml-data-mining-portfolio/
├── notebooks/
│   ├── fetal_health_classification.ipynb
│   ├── rainfall_prediction.ipynb
│   ├── ice_cream_revenue.ipynb
│   ├── car_price_prediction.ipynb
│   ├── customer_segmentation.ipynb
│   ├── market_basket_analysis.ipynb
│   └── titanic_survival.ipynb
├── images/
│   └── [visualization screenshots]
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📫 Contact

Feel free to reach out for collaborations or questions!

**[Your Name]** - [your.email@example.com](mailto:your.email@example.com)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>⭐ If you find this repository helpful, please consider giving it a star!</i>
</p>
