import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Predicciones Ventas Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header-title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    """Cargar el modelo entrenado"""
    return joblib.load(r"D:\forecasting_retailer\model\modelo_final.joblib")

@st.cache_data
def cargar_datos():
    """Cargar los datos de inferencia"""
    return pd.read_csv(r"D:\forecasting_retailer\data\processed\inferencia_df_transformado.csv")

def obtener_columnas_predictoras(df):
    """Obtener columnas para predicción"""
    excluir = ['fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria', 
               'precio_base', 'es_estrella', 'unidades_vendidas', 'precio_venta', 'ingresos',
               'Amazon', 'Decathlon', 'Deporvillage', 'nombre_dia']
    return [col for col in df.columns if col not in excluir and df[col].dtype != 'object']

def actualizar_lags(df_dia, predicciones_dia, producto_id):
    """Actualizar lags para el siguiente día"""
    lag_actual = predicciones_dia
    
    # Desplazar lags
    df_dia['lag7'] = df_dia['lag6'].copy()
    df_dia['lag6'] = df_dia['lag5'].copy()
    df_dia['lag5'] = df_dia['lag4'].copy()
    df_dia['lag4'] = df_dia['lag3'].copy()
    df_dia['lag3'] = df_dia['lag2'].copy()
    df_dia['lag2'] = df_dia['lag1'].copy()
    df_dia['lag1'] = lag_actual
    
    return df_dia

def calcular_media_movil(ultimas_predicciones):
    """Calcular media móvil de 7 días"""
    if len(ultimas_predicciones) < 7:
        return np.mean(ultimas_predicciones) if ultimas_predicciones else 0
    return np.mean(ultimas_predicciones[-7:])

def realizar_predicciones_recursivas(df, modelo, productos_unicos, columnas_pred):
    """Realizar predicciones recursivas para todos los días de noviembre"""
    resultados = []
    predicciones_por_producto = {prod: [] for prod in productos_unicos}
    
    # Días de noviembre
    dias_noviembre = sorted(df['dia_mes'].unique())
    
    for dia in dias_noviembre:
        df_dia = df[df['dia_mes'] == dia].copy()
        
        for producto_id in productos_unicos:
            df_prod = df_dia[df_dia['producto_id'] == producto_id].copy()
            
            if df_prod.empty:
                continue
            
            # Obtener características
            X = df_prod[columnas_pred].values
            
            # Realizar predicción
            pred = modelo.predict(X)[0]
            pred = max(0, pred)  # Asegurar que no sea negativo
            
            # Almacenar resultado
            fecha = pd.to_datetime(df_prod['fecha'].values[0])
            nombre = df_prod['nombre'].values[0]
            categoria = df_prod['categoria'].values[0]
            
            resultados.append({
                'fecha': fecha,
                'dia': dia,
                'producto_id': producto_id,
                'nombre': nombre,
                'categoria': categoria,
                'prediccion': pred
            })
            
            # Guardar predicción para actualizar lags
            predicciones_por_producto[producto_id].append(pred)
        
        # Actualizar lags para el siguiente día
        if dia < max(dias_noviembre):
            df_siguiente = df[df['dia_mes'] == dia + 1].copy()
            
            for producto_id in productos_unicos:
                df_prod_siguiente = df_siguiente[df_siguiente['producto_id'] == producto_id].copy()
                
                if not df_prod_siguiente.empty and producto_id in predicciones_por_producto:
                    pred_actual = predicciones_por_producto[producto_id][-1]
                    
                    # Actualizar lags
                    idx_siguiente = df[df['dia_mes'] == dia + 1].index[0]
                    df.loc[idx_siguiente:df[(df['producto_id'] == producto_id) & (df['dia_mes'] == dia + 1)].index[-1], 'lag1'] = pred_actual
                    
                    # Actualizar media móvil
                    media_movil = calcular_media_movil(predicciones_por_producto[producto_id])
                    df.loc[idx_siguiente:df[(df['producto_id'] == producto_id) & (df['dia_mes'] == dia + 1)].index[-1], 'media_movil_7d'] = media_movil
    
    return pd.DataFrame(resultados)

# Cargar datos
modelo = cargar_modelo()
df_original = cargar_datos()

# Preparar datos
df = df_original.copy()
df['fecha'] = pd.to_datetime(df['fecha'])
columnas_pred = obtener_columnas_predictoras(df)
productos_unicos = df['producto_id'].unique()

# ============== SIDEBAR ==============
st.sidebar.title("⚙️ Panel de Control")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Configuración de Predicción")

# Selector de productos
productos_list = sorted(df['nombre'].unique())
productos_seleccionados = st.sidebar.multiselect(
    "Selecciona productos para visualizar:",
    productos_list,
    default=productos_list[:5]
)

# Categorías
categorias = sorted(df['categoria'].unique())
categoria_filtro = st.sidebar.selectbox(
    "Filtrar por categoría (opcional):",
    ["Todas"] + categorias
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Opciones de Visualización")

tipo_grafico = st.sidebar.radio(
    "Tipo de gráfico principal:",
    ["Línea temporal", "Comparativa por producto", "Distribución"]
)

mostrar_detalles = st.sidebar.checkbox("Mostrar detalles por día", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Información:**\n\n"
    "Las predicciones se calculan de forma recursiva, "
    "actualizando los lags día a día según las predicciones anteriores."
)

# ============== MAIN CONTENT ==============
st.markdown("<h1 class='header-title'>📈 Predicciones de Ventas - Noviembre 2025</h1>", 
            unsafe_allow_html=True)

# Realizar predicciones
with st.spinner("🔄 Calculando predicciones recursivas..."):
    df_predicciones = realizar_predicciones_recursivas(df, modelo, productos_unicos, columnas_pred)

# Filtrar predicciones
if categoria_filtro != "Todas":
    df_predicciones_filtrado = df_predicciones[df_predicciones['categoria'] == categoria_filtro].copy()
else:
    df_predicciones_filtrado = df_predicciones.copy()

df_predicciones_filtrado = df_predicciones_filtrado[
    df_predicciones_filtrado['nombre'].isin(productos_seleccionados)
].copy()

# ============== KPIs ==============
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_unidades = df_predicciones_filtrado['prediccion'].sum()
    st.metric("📦 Total Unidades", f"{total_unidades:,.0f}")

with col2:
    promedio_diario = df_predicciones_filtrado.groupby('dia')['prediccion'].sum().mean()
    st.metric("📅 Promedio Diario", f"{promedio_diario:,.0f}")

with col3:
    max_dia = df_predicciones_filtrado.groupby('dia')['prediccion'].sum().max()
    st.metric("📈 Mejor Día", f"{max_dia:,.0f}")

with col4:
    num_productos = df_predicciones_filtrado['nombre'].nunique()
    st.metric("🏷️ Productos", f"{num_productos}")

st.markdown("---")

# ============== GRÁFICOS ==============
col_grafico_izq, col_grafico_der = st.columns(2)

with col_grafico_izq:
    st.subheader("📊 Predicciones por Día")
    
    # Agrupar por día
    ventas_diarias = df_predicciones_filtrado.groupby('dia')['prediccion'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=ventas_diarias, x='dia', y='prediccion', marker='o', 
                 linewidth=2.5, markersize=8, color='#667eea', ax=ax)
    sns.set_style("whitegrid")
    ax.set_xlabel('Día de Noviembre', fontsize=11, fontweight='bold')
    ax.set_ylabel('Unidades Predichas', fontsize=11, fontweight='bold')
    ax.fill_between(ventas_diarias['dia'], ventas_diarias['prediccion'], alpha=0.2, color='#667eea')
    plt.tight_layout()
    st.pyplot(fig)

with col_grafico_der:
    st.subheader("🏆 Top 5 Productos")
    
    top_productos = df_predicciones_filtrado.groupby('nombre')['prediccion'].sum().nlargest(5)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(y=top_productos.index, x=top_productos.values, palette='husl', ax=ax)
    ax.set_xlabel('Unidades Predichas', fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title('')
    
    for i, v in enumerate(top_productos.values):
        ax.text(v + 5, i, f'{v:,.0f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# ============== GRÁFICOS SECUNDARIOS ==============
col_sec_izq, col_sec_der = st.columns(2)

with col_sec_izq:
    st.subheader("📂 Predicciones por Categoría")
    
    ventas_categoria = df_predicciones_filtrado.groupby('categoria')['prediccion'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(ventas_categoria))
    sns.barplot(data=ventas_categoria, x='categoria', y='prediccion', palette=colors, ax=ax)
    ax.set_xlabel('Categoría', fontsize=11, fontweight='bold')
    ax.set_ylabel('Unidades Predichas', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(ventas_categoria['prediccion'].values):
        ax.text(i, v + 5, f'{v:,.0f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

with col_sec_der:
    st.subheader("📈 Distribución de Predicciones")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_predicciones_filtrado, x='prediccion', bins=30, 
                 kde=True, color='#764ba2', ax=ax)
    ax.set_xlabel('Unidades Predichas', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# ============== TABLA DE DETALLES ==============
if mostrar_detalles:
    st.subheader("📋 Detalles de Predicciones")
    
    # Crear tabla con formato
    df_tabla = df_predicciones_filtrado.copy()
    df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%d/%m/%Y')
    df_tabla['prediccion'] = df_tabla['prediccion'].round(2)
    df_tabla = df_tabla.sort_values(['dia', 'nombre'])
    
    st.dataframe(
        df_tabla[['fecha', 'nombre', 'categoria', 'prediccion']].rename(
            columns={
                'fecha': 'Fecha',
                'nombre': 'Producto',
                'categoria': 'Categoría',
                'prediccion': 'Unidades Predichas'
            }
        ),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ============== ESTADÍSTICAS FINALES ==============
col_stat1, col_stat2, col_stat3 = st.columns(3)

with col_stat1:
    desv_est = df_predicciones_filtrado.groupby('dia')['prediccion'].sum().std()
    st.metric("📊 Desv. Estándar Diaria", f"{desv_est:,.0f}")

with col_stat2:
    coef_variacion = (desv_est / promedio_diario) * 100 if promedio_diario > 0 else 0
    st.metric("📉 Coef. Variación", f"{coef_variacion:.1f}%")

with col_stat3:
    rango = max_dia - df_predicciones_filtrado.groupby('dia')['prediccion'].sum().min()
    st.metric("📏 Rango", f"{rango:,.0f}")

st.markdown("---")
st.caption("🤖 Aplicación de predicciones con HistGradientBoostingRegressor | Noviembre 2025")