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

st.title ("Explorador de videojuegos (1980-2020) 🕵🏼‍♂️​.")

df = pd.read_csv('Data/video_games_sales_completo.csv')

st.dataframe(df)

## Sidebar para filtros ##
st.sidebar.header("Filtrar")
estado_consola = st.sidebar.multiselect("Estado Consola", df["Estado_Consola"].unique())
genero = st.sidebar.multiselect("Genero", df["Genre"].unique())
pegi_categoria = st.sidebar.multiselect("PEGI", df["PEGI_categoria"].unique())
duracion_juego = st.sidebar.multiselect("Duración", df["Duracion_juego_cat"].unique())
precio_min, precio_max = df["Price"].min(), df["Price"].max()
price_range = st.sidebar.slider("Precio Videojuego", precio_min, precio_max, (precio_min, precio_max))


# Aplicar filtros uno a uno (solo si hay selección)
df_filtrado = df.copy()

if estado_consola:
    df_filtrado = df_filtrado[df_filtrado["Estado_Consola"].isin(estado_consola)]

if genero:
    df_filtrado = df_filtrado[df_filtrado["Genre"].isin(genero)]

if pegi_categoria:
    df_filtrado = df_filtrado[df_filtrado["PEGI_categoria"].isin(pegi_categoria)]

if duracion_juego:
    df_filtrado = df_filtrado[df_filtrado["Duracion_juego_cat"].isin(duracion_juego)]

# Filtro de precio (siempre se aplica)
df_filtrado = df_filtrado[df_filtrado["Price"].between(price_range[0], price_range[1])]

st.write (f"Se encontraron {df_filtrado.shape[0]} videojuegos")
st.dataframe(df_filtrado)