#  Predicción de Ventas de Videojuegos

## Overview 

Este proyecto se centra en la materialización de 'MVP' (Mínimo Producto Viable), basado en la creación, desarrollo y uso de un modelo predictivo sobre cuántas ventas va a tener un videojuego si sale al mercado, teniendo en cuenta una serie de características, previamente definidas y una base de datos sobre ventas de videojuegos pasados. 

La interfaz utilizada para que el cliente final interactúe con el modelo es la facilitada por **Streamlit**.

---

## Data sources

- Kaggle [Kaggle](https://www.kaggle.com/datasets/zahidmughal2343/video-games-sale)
- Kaggle [Kaggle](https://www.kaggle.com/datasets/beridzeg45/video-games)

---

##  ¿Qué hace esta aplicación?

Permite al usuario:
- Introducir información sobre un videojuego (género, consola, precio, etc.)
- Obtener una predicción automática de ventas clasificadas (`Ventas_Clase`) en:
  - `Muy Alta`: ventas totales comprendidas entre 1.220.000 y más de 80 millones de copias.
  - `Alta`: ventas totales comprendidas entre 250.000 y 1.210.000 copias.
  - `Normal`: ventas totales comprendidas entre 1.000 y 240.000 copias.

---

##  Modelo utilizado

Se ha entrenado un modelo de **Random Forest Classifier** con un pipeline de preprocesamiento.
Primero, divide el conjunto de datos en entrenamiento y prueba, estratificando por la variable objetivo. Luego, separa las variables numéricas y categóricas para aplicar un preprocesamiento adecuado: escalado estándar para las numéricas (`StandardScaler`) y codificación one-hot para las categóricas (`OneHotEncoder`). 
Finalmente, entrena un modelo de Random Forest con los datos preprocesados y realiza predicciones sobre el conjunto de prueba.

El modelo alcanza una precisión del **70%** con las tres clases previamente descritas.

---

##  Variables independientes utilizadas

- Consola (`Platform`).
- Año de lanzamiento del videojuego (`Year`).
- Género (`Genre`).
- Desarrollador (`Publisher`).
- Precio del juego (`Price`).
- Estado de la consola (`Estado_Consola`).
- Clasificación PEGI (`PEGI_categoria`).
- Modo de juego (`Modo Juego`).
- Duración estimada (`Duracion_juego_cat`).
- Valoracion del usuarios, del 0 al 10 (`User Score`).
- N.º de valoraciones (`User Ratings Count`).
- Precio consola (`Price_Platform`).
- Año de lanzamiento de la consola (`Year_Consola`).
- Situación económica global del año que salio el videojuego (`Situación_Economica`).
- Perteneciente a una Saga de videojuegos o no (`Tipo_Saga`).
- Titulo del Videojuego (`Nombre_Base`).
- Período de tiempo, en años, desde el año que salio al mercado la consola y, el año que salio al mercado el videojuego para esa consola (`Años_desde_lanzamiento_consola`).
- Relación entre el precio del videojuego y el precio de la consola en la que se lanza (`Precio_relativo`).

Las variables ocultas (`Precio_relativo`, `Nombre_Base`, `Años_desde_lanzamiento_consola`) son gestionadas automáticamente por el sistema.

---

##  Resultados del modelo

```text
Random Forest Classification Report:
  Accuracy: 70%
  - Muy Alta: Precision 0.70, Recall 0.38
  - Alta: Precision 0.57, Recall 0.48
  - Normal: Precision 0.75, Recall 0.88
```

---

##  Autor

**Pablo Arrastia Hernández** 