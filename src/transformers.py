import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        agg = X.groupby('CustomerId')['Amount'].agg(
            TotalAmount='sum',
            AvgAmount='mean',
            Count='count',
            StdAmount='std'
        ).fillna(0).reset_index()
        return agg

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['TransHour'] = X[self.datetime_col].dt.hour
        X['TransDay'] = X[self.datetime_col].dt.day
        X['TransMonth'] = X[self.datetime_col].dt.month
        X['TransYear'] = X[self.datetime_col].dt.year
        return X.drop(columns=[self.datetime_col])
