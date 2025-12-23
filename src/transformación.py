import pandas as pd
import holidays

ventas_data = r"D:\forecasting_retailer\data\raw\training\ventas.csv"
competencia_data = r"D:\forecasting_retailer\data\raw\training\competencia.csv"
inferencia_data = r"D:\forecasting_retailer\data\raw\inference\ventas_2025_inferencia.csv"
ventas_df = pd.read_csv(ventas_data)
competencia_df = pd.read_csv(competencia_data)
inferencia_df = pd.read_csv(inferencia_data)

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

def eliminacion_columnas (df):
    df = df.drop(columns=['Amazon', 'Decathlon', 'Deporvillage'])
    return df

def nuevas_columnas (df):
    # Crear copias de las variables categóricas con sufijo _h
    df['nombre_h'] = df['nombre']
    df['categoria_h'] = df['categoria']
    df['subcategoria_h'] = df['subcategoria']
    return df
def dumies (df):
    df = pd.get_dummies(df, columns=['nombre_h', 'categoria_h', 'subcategoria_h'], drop_first=True)
    return df

def fecha_inferencia (inferencia_df):
    inferencia_df['fecha'] = pd.to_datetime(inferencia_df['fecha'])
    inferencia_df['año'] = inferencia_df['fecha'].dt.year
    inferencia_df['mes'] = inferencia_df['fecha'].dt.month
    inferencia_df['dia_mes'] = inferencia_df['fecha'].dt.day
    inferencia_df['dia_semana'] = inferencia_df['fecha'].dt.weekday
    inferencia_df['nombre_dia'] = inferencia_df['fecha'].dt.day_name(locale='es_ES')
    inferencia_df['es_fin_semana'] = inferencia_df['dia_semana'].isin([5, 6])
    return inferencia_df

def festivo_inferencia (inferencia_df):
    festivos_es = []
    for year in inferencia_df['año'].unique():
        festivos_es += [d for d in holidays.country_holidays('ES', years=[year]).keys()]
    inferencia_df['es_festivo'] = inferencia_df['fecha'].isin(festivos_es)
    return inferencia_df

def es_black_friday_inferencia(inferencia_df):
    if inferencia_df.month == 11:
        ultimo_viernes = max([d for d in pd.date_range(start=inferencia_df.replace(day=1), end=inferencia_df.replace(day=30)) if d.weekday() == 4])
        return inferencia_df == ultimo_viernes
    return False

def es_cyber_monday_inferencia(inferencia_df):
    if inferencia_df.month == 11:
        ultimo_viernes = max([d for d in pd.date_range(start=inferencia_df.replace(day=1), end=inferencia_df.replace(day=30)) if d.weekday() == 4])
        cyber_monday = ultimo_viernes + pd.Timedelta(days=3)
        return fecha == cyber_monday
    return False

def columna_inferencia (inferencia_df):
    inferencia_df['es_black_friday'] = inferencia_df['fecha'].apply(es_black_friday)
    inferencia_df['es_cyber_monday'] = inferencia_df['fecha'].apply(es_cyber_monday)
    inferencia_df['semana_anio'] = inferencia_df['fecha'].dt.isocalendar().week
    inferencia_df['trimestre'] = inferencia_df['fecha'].dt.quarter
    inferencia_df['dia_anio'] = inferencia_df['fecha'].dt.dayofyear
    inferencia_df['es_inicio_mes'] = inferencia_df['dia_mes'] == 1
    inferencia_df['es_fin_mes'] = inferencia_df['fecha'].dt.is_month_end
    competencia_cols = ['Amazon', 'Decathlon', 'Deporvillage']
    if all(col in inferencia_df.columns for col in competencia_cols):
        inferencia_df['precio_competencia'] = inferencia_df[competencia_cols].mean(axis=1)
        inferencia_df['ratio_precio'] = inferencia_df['precio_venta'] / inferencia_df['precio_competencia']
    else:
        inferencia_df['precio_competencia'] = np.nan
        inferencia_df['ratio_precio'] = np.nan
    inferencia_df['descuento'] = ((inferencia_df['precio_venta'] - inferencia_df['precio_base']) / inferencia_df['precio_base']) * 100
    return inferencia_df

def lags_inferencia (inferencia_df):
    for lag in range(1, 8):
        inferencia_df[f'lag{lag}'] = inferencia_df.groupby('producto_id')['unidades_vendidas'].shift(lag)
    inferencia_df['media_movil_7d'] = inferencia_df.groupby('producto_id')['unidades_vendidas'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    return inferencia_df

def nuevas_columnas_inferencia (inferencia_df):
    inferencia_df['nombre_h'] = inferencia_df['nombre']
    inferencia_df['categoria_h'] = inferencia_df['categoria']
    inferencia_df['subcategoria_h'] = inferencia_df['subcategoria']
    inferencia_df = pd.get_dummies(inferencia_df, columns=['nombre_h', 'categoria_h', 'subcategoria_h'], drop_first=True)
    inferencia_df = inferencia_df[inferencia_df['mes'] == 11].reset_index(drop=True)
    return inferencia_df