#  Predicción de Ventas de Productos Deportivos.

## Overview 

Este proyecto se centra en la materialización de 'MVP' (Mínimo Producto Viable), basado en la creación, desarrollo y uso de un modelo predictivo sobre la previsión de ventas para saber cuantas unidades de cada producto se van a vender cada día, de noviembre de 2025, incluyendo "BlackFriday".
La interfaz utilizada para que el cliente final interactúe con el modelo es la facilitada por **Streamlit**.

---

##  Modelo utilizado

Se ha entrenado un modelo de **XG Boost** con un pipeline de preprocesamiento.
Primero, divide el conjunto de datos en entrenamiento y prueba, estratificando por la variable objetivo. Luego, separa las variables numéricas y categóricas para aplicar un preprocesamiento adecuado: escalado estándar para las numéricas (`StandardScaler`) y codificación one-hot para las categóricas (`OneHotEncoder`). 
Finalmente, entrena un modelo de XG Boost con los datos preprocesados y realiza predicciones sobre el conjunto de prueba.

---

##  Resultados del modelo (cambiar)

```text
Random Forest Classification Report:
  Accuracy: 70%
  - Muy Alta: Precision 0.70, Recall 0.38
  - Alta: Precision 0.57, Recall 0.48
  - Normal: Precision 0.75, Recall 0.88
```