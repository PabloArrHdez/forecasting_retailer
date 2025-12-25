import streamlit as st
import pandas as pd

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.pexels.com/photos/45056/pexels-photo-45056.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, h5, h6, p, span {{
        color: white !important;
    }}
        section[data-testid="stSidebar"] > div {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    }}
        /* Créditos de la imagen */
    .credit {{
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 13px;
        background-color: rgba(0, 0, 0, 0.6);
        padding: 6px 10px;
        border-radius: 8px;
        color: white;
        border: 1px solid white;
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.4);
        z-index: 100;
    }}
        .credit a {{
        color: white;
        text-decoration: underline;
    }}

    </style>

    <div class="credit">
        Autor: <a href="https://www.pexels.com/es-es/@dom-j-7304/" target="_blank">Dom J</a> en <a href="https://unsplash.com/es" target="_blank">Unsplash</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title ("Explorador de artículos deportivos (2021-2024)​.")

ventas_data = r"D:\forecasting_retailer\data\raw\training\ventas.csv"

ventas_df = pd.read_csv(ventas_data)

st.dataframe(ventas_df)

## Sidebar para filtros ##
st.sidebar.header("Filtrar")
Categoria = st.sidebar.multiselect("categoria", ventas_df["categoria"].unique())
Subcategoria = st.sidebar.multiselect("subcategoria", ventas_df["subcategoria"].unique())
precio_min, precio_max = ventas_df["precio_base"].min(), ventas_df["precio_base"].max()
price_range = st.sidebar.slider("Precio Artículo", precio_min, precio_max, (precio_min, precio_max))

# Aplicar filtros uno a uno (solo si hay selección)
ventas_df_filtrado = ventas_df.copy()

if Categoria:
    ventas_df_filtrado= ventas_df_filtrado[ventas_df_filtrado["categoria"].isin(Categoria)]

if Subcategoria:
    ventas_df_filtrado = ventas_df_filtrado[ventas_df_filtrado["subcateogria"].isin(Subcategoria)]

# Filtro de precio (siempre se aplica)
ventas_df_filtrado = ventas_df_filtrado[ventas_df_filtrado["precio_base"].between(price_range[0], price_range[1])]

st.write (f"Se encontraron {ventas_df_filtrado.shape[0]} artículos deportivos")
st.dataframe(ventas_df_filtrado)