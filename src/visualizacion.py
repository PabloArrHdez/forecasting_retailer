import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def unidades_vendidas_año (df):
    años = df['año'].unique()
    # Creamos un gráfico para cada año
    for año in sorted(años):
        plt.figure(figsize=(16, 5))
        datos_año = df[df['año'] == año]
        sns.lineplot(data=datos_año, x='fecha', y='unidades_vendidas', marker='o', label=f'Año {año}')
        # Marcamos los días de Black Friday
        black_fridays = datos_año[datos_año['es_black_friday']]
        plt.scatter(black_fridays['fecha'], black_fridays['unidades_vendidas'], color='red', label='Black Friday', zorder=5)
        plt.title(f'Unidades vendidas por día en {año} (Black Friday en rojo)')
        plt.xlabel('Fecha')
        plt.ylabel('Unidades vendidas')
        plt.legend()
        plt.tight_layout()
        plt.show();

def unidades_vendidas_semana (df):
    plt.figure(figsize=(10,6))
    orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_dia = df.groupby('nombre_dia')['unidades_vendidas'].sum().reindex(orden_dias)
    sns.barplot(x=df_dia.index, y=df_dia.values, palette='viridis')
    plt.ylabel('Unidades vendidas')
    plt.xlabel('Día de la semana')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show();

def unidades_vendidas_categoria (df):
    plt.figure(figsize=(10,5))
    suma_categoria = df.groupby('categoria')['unidades_vendidas'].sum().sort_values(ascending=False)
    sns.barplot(x=suma_categoria.index, y=suma_categoria.values, palette='crest')
    plt.xlabel('Categoría')
    plt.ylabel('Unidades vendidas')
    plt.title('Suma de unidades vendidas por categoría')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show();

def unidades_vendidas_subcat (df):
    plt.figure(figsize=(12,5))
    suma_subcat = df.groupby('subcategoria')['unidades_vendidas'].sum().sort_values(ascending=False)
    sns.barplot(x=suma_subcat.index, y=suma_subcat.values, palette='mako')
    plt.xlabel('Subcategoría')
    plt.ylabel('Unidades vendidas')
    plt.title('Suma de unidades vendidas por subcategoría')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show();

def top_productos (df):
    plt.figure(figsize=(12,5))
    top_productos = df.groupby('nombre')['unidades_vendidas'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_productos.index, y=top_productos.values, palette='rocket')
    plt.xlabel('Producto')
    plt.ylabel('Unidades vendidas')
    plt.title('Top 10 productos por unidades vendidas')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show();

def densidad_precios (df):
    plt.figure(figsize=(10,6))
    sns.kdeplot(df['precio_venta'], label='Precio venta propio', fill=True, color='blue')
    if 'Amazon' in df.columns:
        sns.kdeplot(df['Amazon'], label='Precio Amazon', fill=True, color='orange')
    plt.xlabel('Precio')
    plt.ylabel('Densidad')
    plt.title('Distribución de precios: propio vs Amazon')
    plt.legend()
    plt.tight_layout()
    plt.show();