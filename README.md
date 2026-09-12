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

- Skincare is the top revenue generator.
- Supplier 5 has the highest defect rate, followed by Supplier 3 and Supplier 2; Supplier 1 is best at ~1.80%.
- Route B is the biggest bottleneck with ~18.2 days average lead time.
- Air is the most expensive transport mode; Sea is cheapest.
- Mumbai and Kolkata generate the most revenue; Delhi the least.
- Random Forest risk model: lead time is the top driver of high-risk shipments, followed by price, stock levels, shipping costs, and availability.

## Technologies

- Python - Core language
- Pandas & NumPy - Data manipulation
- Matplotlib & Seaborn - Visualization
- Scikit-learn - Machine learning (Random Forest)
