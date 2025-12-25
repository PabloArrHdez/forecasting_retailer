import streamlit as st

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://www.pexels.com/es-es/foto/campo-de-pista-marron-y-blanco-163444/");
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
st.title("Descripción de los datos ✍️.​")
st.markdown("""
Este conjunto de datos, previo a una posterior limpieza, unión y transformación de los mismos, simula las características que influyen en las ventas globales de un videojuego en el año que sale al mercado.
            """)