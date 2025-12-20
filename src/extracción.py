import pandas as pd

ventas_data = r"D:\forecasting_retailer\data\raw\training\ventas.csv"
competencia_data = r"D:\forecasting_retailer\data\raw\training\competencia.csv"

ventas_df = pd.read_csv(ventas_data)
competencia_df = pd.read_csv(competencia_data)

