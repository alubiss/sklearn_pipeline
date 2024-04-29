import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer: TransformerMixin, input_cols:list = None):
        """wrapper dla obiektów sklearn dodający wsparcie dla Pandas DataFrame"""
        self.input_cols = input_cols
        self.transformer = transformer
              
    def fit(self, X: pd.DataFrame, y=None, **fit_params) -> "DataFrameTransformer":
        if self.input_cols is None:
            self.input_cols = X.columns
        self._base_cols_names = X.columns
        self.transformer.fit(X[self.input_cols], y, **fit_params)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _X = X.copy()
        transform = self.transform(_X[self.input_cols])
        df_tran = pd.DataFrame(transform, columns=self.input_cols, index= X.index)
        result = pd.concat([_X.drop(columns=self.input_cols), df_trans], axis=1)
        return result[self._base_cols_names]
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self,input_cols:list):
        """usuwanie zbednych kolumn"""
        self.input_cols = input_cols

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.input_cols)