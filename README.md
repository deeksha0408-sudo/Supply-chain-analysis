# Supply Chain Analysis

End-to-end data analysis and machine learning project on a supply chain dataset using Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn).

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Dataset Overview](#2-dataset-overview)
- [3. Table Structure](#3-table-structure)
- [4. Data Import Process](#4-data-import-process)
- [5. Data Cleaning](#5-data-cleaning)
- [6. Exploratory Data Analysis](#6-exploratory-data-analysis)
- [7. Business Insights](#7-business-insights)
- [8. Conclusion](#8-conclusion)
- [9. Screenshots and other information](#9-screenshots-and-other-information)

## 1. Introduction

This project performs a complete supply chain analysis, moving from raw data to actionable business insights. It explores revenue generation, shipping costs, supplier performance, inventory levels, route bottlenecks, and defect rates across the supply chain. It then trains a **Random Forest classifier** to predict high-risk products (those with above-median defect rates) and identifies the most influential risk features.

The project is structured as a step-by-step Jupyter notebook (`supply_chain_analysis.ipynb`) covering data loading, cleaning, exploratory analysis, visualization, feature engineering, and model building.

## 2. Dataset Overview

- **Source file:** `dataset/supply_chain_data.csv`
- **Total rows:** 100
- **Total columns:** 24
- **File size:** Small (~100-record retail supply chain sample)
- **Rows/Columns in notebook output:** `(100, 24)`

The dataset contains transactional and operational records spanning product types (skincare, haircare, cosmetics), suppliers, locations (Mumbai, Kolkata, etc.), shipping carriers, transportation modes, and routes.

## 3. Table Structure

The dataset contains the following columns:

| Column | Description |
|--------|-------------|
| `Product type` | Category of the product (skincare, haircare, cosmetics) |
| `SKU` | Stock Keeping Unit identifier |
| `Price` | Product selling price |
| `Availability` | Product availability percentage/days |
| `Number of products sold` | Units sold |
| `Revenue generated` | Total revenue from the product |
| `Customer demographics` | Target customer segment |
| `Stock levels` | Current stock quantity |
| `Lead times` | Order-to-delivery lead time (days) |
| `Order quantities` | Order size |
| `Shipping times` | Number of shipping days |
| `Shipping carriers` | Carrier used (Carrier A/B/C) |
| `Shipping costs` | Cost of shipping |
| `Supplier name` | Supplier identifier (Supplier 1–5) |
| `Location` | Supplier location/city |
| `Lead time` | Supplier lead time (days) |
| `Production volumes` | Quantity produced |
| `Manufacturing lead time` | Time to manufacture |
| `Manufacturing costs` | Cost of manufacturing |
| `Inspection results` | Quality inspection outcome (Pending, Fail, Pass) |
| `Defect rates` | Product defect rate |
| `Transportation modes` | Mode of transport (Road, Air, Rail, Sea) |
| `Routes` | Shipping route identifier (Route A/B/C...) |
| `Costs` | Total associated costs |

## 4. Data Import Process

The notebook starts by importing the required libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

The dataset is loaded using Pandas:

```python
df = pd.read_csv("supply_chain_data.csv")
```

Then the shape and a preview are printed:

```python
print(df.shape)   # (100, 24)
df.head()
```

## 5. Data Cleaning

The cleaning step runs as follows:

```python
# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(exclude='object'):
    df[col].fillna(df[col].median(), inplace=True)
```

- **Duplicates:** Removed with `drop_duplicates()`.
- **Categorical missing values:** Filled with the **mode** (most frequent value).
- **Numerical missing values:** Filled with the **median** (robust to outliers).

Data quality checks (`df.info()`, `df.describe()`, `df.isnull().sum()`) are run before cleaning to understand types, distributions, and nulls.

## 6. Exploratory Data Analysis

The EDA covers the following analyses with visualizations:

### 6.1 Revenue Analysis
Bar chart of total `Revenue generated` grouped by `Product type`, showing which category drives the most revenue.

### 6.2 Shipping Cost Analysis
Box plot of `Shipping costs` by `Transportation modes` to compare cost distribution across Road, Air, Rail, and Sea.

### 6.3 Supplier Performance
Aggregation by `Supplier name` (mean `Lead time`, mean `Defect rates`, total `Revenue generated`), sorted to surface poorly performing suppliers.

### 6.4 Inventory Analysis
Histogram (with KDE) of `Stock levels` to inspect the inventory distribution.

### 6.5 Bottleneck Detection
Route-level aggregation of mean `Lead time` and `Shipping costs`, sorted descending by lead time to find the slowest routes.

### 6.6 Root Cause Analysis
Mean `Defect rate` per supplier, plotted as a bar chart to identify quality problem areas.

## 7. Business Insights

The notebook computes the following actionable insights:

- **Highest revenue product category:** The product type with the maximum total revenue (printed with its dollar value).
- **Supplier with highest defect rate:** Supplier with the worst average defect rate, flagged for quality improvement.
- **Route with maximum lead time:** The shipping route with the highest average lead time, indicating a bottleneck.

### Machine Learning Model

- **Target feature engineering:** A binary `High_Risk` flag is created where `1` = defect rate above the median:

```python
df['High_Risk'] = np.where(
    df['Defect rates'] > df['Defect rates'].median(), 1, 0
)
```

- **Model:** `RandomForestClassifier` (100 trees, `random_state=42`).
- **Features used:** `Price`, `Availability`, `Stock levels`, `Lead time`, `Shipping costs`.
- **Split:** 80/20 train-test split with `random_state=42`.
- **Evaluation:** `classification_report` plus a confusion matrix heatmap.
- **Feature importance:** A bar chart of each feature's importance ranking, showing which variables most influence the high-risk prediction.

## 8. Conclusion

The project demonstrates a complete supply chain analytics workflow:

1. Loading and cleaning a raw supply chain dataset.
2. Profiling revenue, shipping, inventory, and supplier quality.
3. Detecting delays and bottlenecks by route and supplier.
4. Building a Random Forest model to predict high-risk (high-defect) products.
5. Interpreting feature importance to guide operational decisions.

The insights enable data-driven decisions such as renegotiating with low-quality suppliers, optimizing shipping routes, and prioritizing inventory for high-revenue product lines.

## 9. Screenshots and other information

Screenshots of the analysis outputs — including the revenue bar chart, shipping cost box plot, stock level histogram, defect rate bar chart, confusion matrix, and feature importance plot — are produced when the notebook is run and can be added to a `screenshots/` folder.

### Project Structure

```
supply-chain-analysis/
├── supply_chain_analysis.ipynb   # Main analysis notebook
├── dataset/
│   └── supply_chain_data.csv      # Raw dataset
└── README.md                      # Project documentation
```

### Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### How to Run

1. Clone the repository.
2. Place/keep `supply_chain_data.csv` inside the `dataset/` folder (or the notebook root, matching the notebook's `pd.read_csv("supply_chain_data.csv")` call).
3. Open `supply_chain_analysis.ipynb` in Jupyter Notebook/Lab or VS Code.
4. Run all cells.

---

**Author:** deeksha0408-sudo