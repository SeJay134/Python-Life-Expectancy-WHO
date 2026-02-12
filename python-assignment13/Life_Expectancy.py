import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.express as px

# Setup
st.set_page_config(
    page_title="Life Expectancy (WHO)",
    page_icon="🧬",
    layout="wide"
)

# The way to the current scripts
BASE_DIR = Path(__file__)

# Read CSV
csv_path = BASE_DIR.parent / "db" / "Life-Expectancy-Data.csv"

# Data
db = pd.read_csv(csv_path)
# print(" \n", db.head())
# print()

# The Life Expectancy (WHO) dataset was loaded directly from Kaggle. 
# The dataset contains country-level health, economic, and demographic indicators collected over multiple years.

# checking
print("--columns--")
print(db.columns)
print()

print("--info--")
db.info()
print()

# No additional data type conversion was required, as all numerical and categorical
# features were already stored in appropriate formats.


# Data cleaning

print("--describe--\n", db.describe())
print()
for col in db:
    print("--describe--\n", db[col].describe().tail(2)) # max vs 75%

# The descriptive statistics reveal a wide range of values across several features. 
# For example, variables such as infant deaths, measles cases, and GDP show large differences 
# between the 75th percentile and maximum values, indicating the presence of potential outliers.

print("--nan--")
print(db.isna().sum())
print()
# Several numerical features contain missing values. 
# Columns such as Population, Hepatitis B, GDP, and Schooling have a relatively high proportion of missing 
# observations and require imputation.

print('--duplicated--\n', db.duplicated().sum())
# No duplicated rows were found in the dataset.
# # The descriptive statistics revealed missing values across multiple numerical features, 
# as well as a wide range of values in variables such as GDP and Adult Mortality, 
# indicating potential outliers. Some features, including Schooling and 
# Income composition of resources, contained zero values which may represent either true 
# measurements or missing data encoded as zeros.

print("Unique countries")
print(sorted(db['Country'].unique()))
print()
# The list of unique country names was inspected to identify inconsistencies or 
# multiple representations of the same country.
# Although some countries appear under their formal WHO designations 
# (e.g., “United Kingdom of Great Britain and Northern Ireland”, “Iran (Islamic Republic of)”), 
#  no duplicate naming formats were found.
# Each country appears only once in a consistent format. Therefore, no country name standardization 
# was required.

# Original dataset was preserved before cleaning steps were applied
db_clean = db.copy()
#print('--db_clean--\n', db_clean.head())
db_clean.columns = db_clean.columns.str.strip()
#print('db_clean.columns\n', db_clean.columns)
# print()
# Column names were stripped of leading and trailing whitespace to ensure
# consistent column referencing.

# 
nan_percent = (db_clean.isna().sum() / len(db_clean) * 100).round(1)
print("nan_percent.sort_values\n", nan_percent.sort_values(ascending=False))
print()
# The percentage of missing values varies across features.
# Imputation strategies were selected based on the proportion of missing data
# in each column.

# Missing values were handled using two strategies:
# Columns with a significant proportion of missing values (Population, Hepatitis B, GDP, Total expenditure, Alcohol, Income composition of resources, Schooling) were imputed using the median.
# Columns with few missing values (<2%) were filled using simple imputation (fillna), preserving the overall distribution.
# Columns with no missing values were left unchanged.

for col in db_clean.columns:
    if not pd.api.types.is_numeric_dtype(db_clean[col]): # int type
        continue
    if 0 < nan_percent[col] <= 1.2:
        db_clean[col] = db_clean[col].fillna(db_clean[col].mean())
    
    if 1.2 < nan_percent[col] <= 50:
        db_clean[col] = db_clean[col].fillna(db_clean[col].median())

# Missing values were handled using two strategies:
# - Columns with a small proportion of missing values (≤ 1.2%) were imputed
#   using the mean.
# - Columns with a higher proportion of missing values (> 1.2% and ≤ 50%)
#   were imputed using the median to reduce the influence of outliers.


# Feature Engineering

# Additional features were created to better capture relationships between
# health, economic, and demographic factors and life expectancy.

# Life Expectancy Difference
db_clean['Life Expectancy Difference'] = db_clean.groupby('Country')['Life expectancy'].transform(lambda x: x.max() - x.min())
# Life expectancy vs Adult Mortality & Alcohol
db_clean["Mortality Alcohol Index"] = db_clean["Adult Mortality"] * db_clean["Alcohol"]
# Total expenditure vs Population
db_clean["Health Expenditure Per Capita"] = db_clean["Total expenditure"] / (db_clean["Population"] / 1000000)

print("info")
db_clean.info()
print()


# Aggregation

# Global life-expectancy trend
db_clean_globaltrend = (db_clean.groupby('Year')['Life expectancy'].mean())
print("db_clean_globaltrend\n", db_clean_globaltrend)
## Global Life Expectancy Trend
# By aggregating life expectancy by year, we can observe a clear upward global trend,
# indicating overall improvements in healthcare, living standards, and disease prevention worldwide.

# Trends for developing vs developed
db_clean_devtrend = (db_clean.groupby(['Year', 'Status'])['Life expectancy'].mean().reset_index())
print("db_clean_devtrend\n", db_clean_devtrend)
print()

### Life Expectancy Trends by Country Status (2000–2015)

# This table shows the **average life expectancy** for Developed and Developing countries 
# over 15 years.  

# **Observations:**  
# - Developed countries consistently have higher life expectancy than Developing countries.  
# - The gap between Developed and Developing countries remains fairly stable over time.  
# - There is a slight upward trend for both groups, indicating gradual improvements in health 
# over the period.

# Stability of Country Status Over Time
db_clean_count_country = db_clean.groupby(['Year', 'Status'])['Country'].nunique()
print("db_clean_count_country", db_clean_count_country)
print()

# When analyzing the dataset from 2000 to 2015, it is noticeable that the number of countries 
# classified as Developed and Developing remains almost constant over the 15-year period.

# Developed countries consistently number around 32.
# Developing countries consistently number around 151–161.
# This indicates that, within this dataset, no countries changed their status from Developing 
# to Developed during these years.

# Interpretation:
# This pattern may reflect the dataset’s classification criteria rather than real-world economic 
# transitions. While it highlights a limitation in observing status changes over time, 
# it still allows us to analyze other factors, such as life expectancy, mortality, 
# and healthcare indicators, across these two groups of countries.

# Life expectancy in countries, Adult Mortality vs Alcohol
db_clean_adultvsalcohol = db_clean.groupby('Country')[['Adult Mortality', 'Alcohol']].mean().reset_index()
print("db_clean_adultvsalcohol", db_clean_adultvsalcohol)
# The dataset was grouped by country to calculate the average Adult Mortality and 
# Alcohol consumption for each country. 

# Life expectancy deciles (10 buckets) vs various driving factors
db_clean['GDP Decile'] = pd.qcut(db_clean["GDP"], q=10, labels=False)
db_clean_GDP_Decile_trend = db_clean.groupby('GDP Decile')['Life expectancy'].mean().reset_index()

