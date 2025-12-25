import streamlit as st
import pandas as pd

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.pexels.com/photos/45056/pexels-photo-45056.jpeg?auto=compress&cs=tinysrgb&w=1280");
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
        Autor: <a href="https://unsplash.com/es/@benofthenorth" target="_blank">Ben Griffiths</a> en <a href="https://unsplash.com/es" target="_blank">Unsplash</a>
    </div>
    """,
    unsafe_allow_html=True
)