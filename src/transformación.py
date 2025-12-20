import pandas as pd

ventas_data = r"D:\forecasting_retailer\data\raw\training\ventas.csv"
competencia_data = r"D:\forecasting_retailer\data\raw\training\competencia.csv"
ventas_df = pd.read_csv(ventas_data)
competencia_df = pd.read_csv(competencia_data)

def fecha (ventas_df,competencia_df):
    ventas_df['fecha'] = pd.to_datetime(ventas_df['fecha'])
    competencia_df['fecha'] = pd.to_datetime(competencia_df['fecha'])
    return ventas_df, competencia_df

def union (ventas_df, competencia_df):
    df = pd.merge(
    ventas_df,
    competencia_df,
    how='inner',
    on=['fecha', 'producto_id']
    )
    return df