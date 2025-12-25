import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Estimador")

# CSS para fondo
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.pexels.com/photos/163444/sport-treadmill-tor-route-163444.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: white !important;
    }}
        section[data-testid="stSidebar"] > div {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    }}
    /* Estilo para los tabs */
    div[data-testid="stTabs"] > div {{
        background-color: rgba(0, 0, 0, 1.0);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid white;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
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
        Autor: <a href="https://www.pexels.com/es-es/@pixabay/" target="_blank">pixabay</a> en <a href="https://www.pexels.com/es-es/" target="_blank">Pexels</a>
    </div>
    """,
    unsafe_allow_html=True
)


st.title('🎮 Modelo predictivo categórico de ventas de Videojuegos. 👾')

tab1, tab2 = st.tabs(["Contacto", "Resumen"])

with tab1:
    st.subheader("Contacto")
    st.write("Autor: Pablo Arrastia Hernández")
    st.write("Email: pabloarrhdez@gmail.com")
    st.write("Cuenta GitHub: https://github.com/PabloArrHdez")


with tab2:
    st.subheader("Resumen del proyecto")
    st.markdown(
    """
    A continuación mostramos un modelo predictivo **Random Forest** donde, gracias a las características propias que influyen en la venta de un videojuego, puede predecir, con un 70% de acierto, en qué intervalo categórico de cantidad de unidades vendidas se encuentra el videojuego que se prediga.

    El resultado que muestra el modelo se divide en tres categorías de estimación de ventas:

    - **Normal**: ventas comprendidas entre 1.000 y 240.000 unidades.  
    - **Alta**: ventas comprendidas entre 250.000 y 1.210.000 unidades.  
    - **Muy Alta**: ventas comprendidas entre 1.220.000 y más de 80 millones.
    """
)