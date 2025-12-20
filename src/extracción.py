import pandas as pd

ventas_data = r"D:\forecasting_retailer\data\raw\training\ventas.csv"
competencia_data = r"D:\forecasting_retailer\data\raw\training\competencia.csv"

ventas_df = pd.read_csv(ventas_data)
competencia_df = pd.read_csv(competencia_data)

def validacion_datos (ventas_df, competencia_df):
    print("Tipos de variables")
    print(ventas_df.dtypes)
    print("\n")
    print(competencia_df.dtypes)
    print("\n")

def nulos (ventas_df, competencia_df):
    print("Valores nulos por columna")
    print(ventas_df.isnull().sum())
    print("\n")
    print(competencia_df.isnull().sum())
    print("\n")

def duplicados (ventas_df, competencia_df):
    duplicados = ventas_df.duplicated().sum()
    print(f"Filas duplicadas: {duplicados}\n")
    duplicados_c = competencia_df.duplicated().sum()
    print(f"Filas duplicadas: {duplicados_c}\n")

def descriptivo (ventas_df, competencia_df):
    print ("Estadistica descriptiva")
    print(ventas_df.describe(include='all'))
    print("\n")
    print(competencia_df.describe(include='all'))
    print("\n")

def resumen_final (ventas_df, competencia_df):
    print ("Resumen final")
    if ventas_df.isnull().sum().sum() == 0:
        print("No hay valores nulos")
    else:
        print("Existen valores nulos")
    if duplicados == 0:
        print("No hay filas duplicadas")
    else:
        print(f"Hay {duplicados} filas duplicadas")
    print("\n")
    if competencia_df.isnull().sum().sum() == 0:
        print("No hay valores nulos")
    else:
        print("Existen valores nulos")
    if duplicados == 0:
        print("No hay filas duplicadas")
    else:
        print(f"Hay {duplicados} filas duplicadas")