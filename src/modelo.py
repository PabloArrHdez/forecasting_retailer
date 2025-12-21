import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

df_data = r"D:\forecasting_retailer\data\processed\df.csv"
df = pd.read_csv(df_data)

def entrenamiento (df):
    train_df = df[df['año'].isin([2021, 2022, 2023])].copy()
    validation_df = df[df['año'] == 2024].copy()
    print(f"Registros en train_df: {len(train_df)}")
    print(f"Registros en validation_df: {len(validation_df)}")

def estimador (df):
    class ColumnSelector(BaseEstimator, TransformerMixin):
        """Selecciona columnas específicas excluyendo las no deseadas."""
    
    def __init__(self, exclude_cols=None, exclude_dtypes=None):
        self.exclude_cols = exclude_cols or []
        self.exclude_dtypes = exclude_dtypes or []
        
    def fit(self, X, y=None):
        # Identifica las columnas a mantener
        self.feature_names_ = [
            col for col in X.columns 
            if col not in self.exclude_cols 
            and X[col].dtype not in self.exclude_dtypes
        ]
        return self
    
    def transform(self, X):
        return X[self.feature_names_]
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
    
def Pipeline (df):
    pipeline = Pipeline([
        ('selector', ColumnSelector(
        exclude_cols=['fecha', 'ingresos', 'unidades_vendidas'],
        exclude_dtypes=['O']  # Excluye tipo object
        )),
        ('scaler', StandardScaler()),  # Opcional: normalización
        ('model', HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        max_depth=7,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42
        ))
    ])
    # Entrenamiento
    pipeline.fit(train_df, train_df['unidades_vendidas'])
    # Predicciones
    y_pred = pipeline.predict(validation_df)
    # Baseline naive (referencia)
    y_pred_naive = np.full(len(validation_df), train_df['unidades_vendidas'].mean())