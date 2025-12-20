import pandas as pd
import holidays

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

def es_black_friday(fecha):
    if fecha.month == 11:
        # Encuentra el último viernes de noviembre
        ultimo_viernes = max([d for d in pd.date_range(start=fecha.replace(day=1), end=fecha.replace(day=30)) if d.weekday() == 4])
        return fecha == ultimo_viernes
    return False

def es_cyber_monday(fecha):
    if fecha.month == 11 or fecha.month == 12:
        # Encuentra el último viernes de noviembre
        ultimo_viernes = max([d for d in pd.date_range(start=fecha.replace(day=1), end=fecha.replace(day=30)) if d.weekday() == 4])
        cyber_monday = ultimo_viernes + pd.Timedelta(days=3)
        return fecha == cyber_monday
    return False

def columnnas (df):
    # Definir festivos en España
    festivos_es = holidays.country_holidays('ES', years=df['fecha'].dt.year.unique())
    # Año
    df['año'] = df['fecha'].dt.year
    # Mes
    df['mes'] = df['fecha'].dt.month
    # Día del mes
    df['dia_mes'] = df['fecha'].dt.day
    # Día de la semana (0=Lunes, 6=Domingo)
    df['dia_semana'] = df['fecha'].dt.weekday
    # Nombre del día de la semana
    df['nombre_dia'] = df['fecha'].dt.day_name(locale='es_ES') if hasattr(df['fecha'].dt, 'day_name') else df['fecha'].dt.dayofweek
    # Es fin de semana
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6])
    # Es festivo nacional
    df['es_festivo'] = df['fecha'].isin(festivos_es)
    # Back Friday #
    df['es_black_friday'] = df['fecha'].apply(es_black_friday)
    df['es_cyber_monday'] = df['fecha'].apply(es_cyber_monday)
    # Semana del año
    df['semana_año'] = df['fecha'].dt.isocalendar().week
    # Trimestre
    df['trimestre'] = df['fecha'].dt.quarter
    # Día del año
    df['dia_año'] = df['fecha'].dt.dayofyear
    # Es inicio de mes
    df['es_inicio_mes'] = df['dia_mes'] == 1
    # Es fin de mes
    df['es_fin_mes'] = df['fecha'].dt.is_month_end
    return df

def lags (df):
    for producto in df['producto_id'].unique():
        for anio in df[df['producto_id'] == producto]['año'].unique():
            mask = (df['producto_id'] == producto) & (df['año'] == anio)
            df.loc[mask, 'lag1'] = df.loc[mask, 'unidades_vendidas'].shift(1)
            df.loc[mask, 'lag2'] = df.loc[mask, 'unidades_vendidas'].shift(2)
            df.loc[mask, 'lag3'] = df.loc[mask, 'unidades_vendidas'].shift(3)
            df.loc[mask, 'lag4'] = df.loc[mask, 'unidades_vendidas'].shift(4)
            df.loc[mask, 'lag5'] = df.loc[mask, 'unidades_vendidas'].shift(5)
            df.loc[mask, 'lag6'] = df.loc[mask, 'unidades_vendidas'].shift(6)
            df.loc[mask, 'lag7'] = df.loc[mask, 'unidades_vendidas'].shift(7)
            df.loc[mask, 'media_movil_7d'] = df.loc[mask, 'unidades_vendidas'].rolling(window=7, min_periods=1).mean()
    # Eliminar registros con nulos en los nuevos lags o media móvil
    df = df.dropna(subset=['lag1','lag2','lag3','lag4','lag5','lag6','lag7','media_movil_7d']).reset_index(drop=True)
    return df

def descuento (df):
    # Crear variable de descuento en porcentaje
    df['descuento'] = ((df['precio_venta'] - df['precio_base']) / df['precio_base']) * 100
    return df

def calculos (df):
    # Calcular el precio promedio de la competencia
    df['precio_competencia'] = df[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    # Calcular el ratio de nuestro precio respecto a la competencia
    df['ratio_precio'] = df['precio_venta'] / df['precio_competencia']
    return df