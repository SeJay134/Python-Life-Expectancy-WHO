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
# print("--columns--")
# print(db.columns)
# print()

# print("--info--")
# db.info()
# print()

# No additional data type conversion was required, as all numerical and categorical
# features were already stored in appropriate formats.