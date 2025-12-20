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