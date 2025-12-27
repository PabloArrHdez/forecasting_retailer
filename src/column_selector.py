from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selecciona columnas específicas excluyendo las no deseadas."""
    def __init__(self, exclude_cols=None, exclude_dtypes=None):
        self.exclude_cols = exclude_cols or []
        self.exclude_dtypes = exclude_dtypes or []

    def fit(self, X, y=None):
        self.feature_names_ = [
            col for col in X.columns 
            if col not in self.exclude_cols 
            and X[col].dtype.name not in self.exclude_dtypes
        ]
        return self

    def transform(self, X):
        if hasattr(X, 'columns'):
            print("Esperadas:", self.feature_names_)
            print("X.columns:", list(X.columns))
            for col in self.feature_names_:
                assert col in X.columns, f"Falta la columna: {col}"
            return X[self.feature_names_]
        else:
            print("X es un array de numpy, columnas esperadas:", self.feature_names_)
            return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
