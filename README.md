# 📦 Supply Chain Analysis & Risk Prediction

An exploratory data analysis and machine learning project on supply chain operations using Python to uncover insights on revenue, shipping costs, supplier performance, inventory, and defect risk.

## Analysis Overview

- Data Import & Cleaning
- Exploratory Data Analysis (EDA)
- Revenue & Shipping Cost Analysis
- Supplier Performance & Bottleneck Detection
- Risk Prediction with Random Forest

## Data Exploration

- Dataset shape check (100 rows, 24 columns)
- Data type and null value check across all columns
- Descriptive statistics for numerical fields
- Duplicate removal
- Missing value handling (mode for categorical, median for numerical)

## Data Validation

- Revenue distribution across product types
- Shipping cost distribution by transport mode
- Stock level distribution
- Supplier lead time and defect rate summary

## Insights

- Skincare is the top revenue generator (~$241.6k) vs haircare (~$174.5k) and cosmetics (~$161.5k).
- Supplier 5 has the highest defect rate (2.67%), followed by Supplier 3 (2.47%) and Supplier 2 (2.36%); Supplier 1 is best at ~1.80%.
- Route B is the biggest bottleneck with ~18.2 days average lead time.
- Air is the most expensive transport mode (~$6.02 avg shipping cost); Sea is cheapest (~$4.97).
- Mumbai and Kolkata generate the most revenue; Delhi the least.
- Random Forest risk model: lead time is the top driver of high-risk shipments (importance 0.31), followed by price, stock levels, shipping costs, and availability.

- High-risk product prediction using a Random Forest Classifier
- Feature importance ranking for risk prediction
- Top suppliers and routes to watch for operational improvement

## Technologies

- Python - Core language
- Pandas & NumPy - Data manipulation
- Matplotlib & Seaborn - Visualization
- Scikit-learn - Machine learning (Random Forest)

## About

Supply Chain Analysis & Risk Prediction