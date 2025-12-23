import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import holidays
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import src.extracción as ext
import src.modelo as mdl
import src.transformación as trs
import src.visualizacion as vsl


ventas_data = r"D:\forecasting_retailer\data\raw\training\ventas.csv"
competencia_data = r"D:\forecasting_retailer\data\raw\training\competencia.csv"
inferencia_data = r"D:\forecasting_retailer\data\raw\inference\ventas_2025_inferencia.csv"
ventas_df = pd.read_csv(ventas_data)
competencia_df = pd.read_csv(competencia_data)
inferencia_df = pd.read_csv(inferencia_data)