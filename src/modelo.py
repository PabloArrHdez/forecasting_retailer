import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

df_data = r"D:\forecasting_retailer\data\processed\df.csv"
df = pd.read_csv(df_data)


train_df = df[df['año'].isin([2021, 2022, 2023])].copy()
validation_df = df[df['año'] == 2024].copy()

print(f"Registros en train_df: {len(train_df)}")
print(f"Registros en validation_df: {len(validation_df)}")