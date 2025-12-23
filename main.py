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

ext.validacion_datos (ventas_df, competencia_df)
ext.nulos (ventas_df, competencia_df)
ext.duplicados (ventas_df, competencia_df)
ext.descriptivo (ventas_df, competencia_df)
ext.resumen_final (ventas_df, competencia_df)

trs.fecha (ventas_df,competencia_df)
trs.union (ventas_df, competencia_df)
trs.es_black_friday(fecha)
trs.es_cyber_monday(fecha)
trs.columnnas(df)
trs.lags(df)
trs.descuento(df)
trs.eliminacion_columnas(df)
trs.nuevas_columnas(df)
trs.dumies(df)
trs.fecha_inferencia(inferencia_df)
trs.festivo_inferencia(inferencia_df)
trs.es_black_friday_inferencia(inferencia_df)
trs.es_cyber_monday_inferencia(inferencia_df)
trs.columna_inferencia(inferencia_df)
trs.lags_inferencia(inferencia_df)
trs.nuevas_columnas_inferencia(inferencia_df)

vsl.unidades_vendidas_año (df)
vsl.unidades_vendidas_semana (df)
vsl.unidades_vendidas_categoria (df)
vsl.unidades_vendidas_subcat (df)
vsl.top_productos (df)
vsl.densidad_precios (df)
vsl.prediccion_realidad (df)

