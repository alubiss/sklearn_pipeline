import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from datetime import datetime, timedelta

class MapOfferValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._banner_mapping = dict()
        self._products_offered_mapping = dict()
        self._cta_mapping = dict()
    
    def fit(self, X , y=None):
        unique_ids = X['id'].unique()
        self._banner_mapping = dict(zip(unique_ids, X.loc[X['id'].isin(unique_ids), 'is_banner']))
        self._products_offered_mapping = dict(zip(unique_ids, X.loc[X['id'].isin(unique_ids), 'products_offered']))
        self._cta_mapping = dict(zip(unique_ids, X.loc[X['id'].isin(unique_ids), 'cta_number']))
        return self
    
    def transform(self, X):
        df = X.copy()
        df['id'] = df['offer_id']
        df['is_banner'] = df['id'].map(self._banner_mapping)
        df['products_offered'] = df['id'].map(self._products_offered_mapping)
        df['cta_number'] = df['id'].map(self._cta_mapping)
        return df

class BannerEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        col = 'is_banner'
        df[f"{col}_cat"] = df[col].map({'False': 0, 'True': 1})
        df = df.drop(columns=[col])
        return df
    
class DataTranform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """cechy dotyczące daty"""
        df = X.copy()
        col ='request_dttm'
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
        df["data_year"]= df[col].dt.year
        df["data_month"]= df[col].dt.month
        df["data_day"]= df[col].dt.day
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
        df['days_from_today'] = (df[col] - today).dt.days
        #df = df.drop(columns=col)
        return df
    
class ValEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        col = 'context.booking.param5'
        df[col] = df[col].clip(0,10)
        return df
    
class TrigonometricTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        month = 'data_month'
        day = 'data_day'
       
        df[f"{month}_SIN"] = df[month].apply(lambda x: np.sin(x * 2 * np.pi /12))
        df[f"{month}_COS"] = df[month].apply(lambda x: np.cos(x * 2 * np.pi /12))
        
        df[f"{day}_SIN"] = df[day].apply(lambda x: np.sin(x * 2 * np.pi /31))
        df[f"{day}_COS"] = df[day].apply(lambda x: np.cos(x * 2 * np.pi /31))
        
        return df 
    
class ColTypeFromater(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['context.booking.param4'] = df['context.booking.param4'].astype(int)
        df['context.leg.param1'] = df['context.leg.param1'].astype(int)
        df['context.leg.param2'] = df['context.leg.param2'].astype(int)
        df['context.leg.param3'] = df['context.leg.param3'].astype(int)
        df['context.leg.param4'] = df['context.leg.param4'].astype(int)
        df['context.leg.param6'] = df['context.leg.param6'].astype(int)
        df['context.param1'] = df['context.param1'].astype(int)
        df['products_offered'] = df['products_offered'].astype(int)
        df['cta_number'] = df['cta_number'].astype(int)
        return df
    
    
class ThreeMonthsAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._N_click_3m_mapping = dict()
        self._udzial_3m_mapping = dict()
    
    def fit(self, X , y=None):
        df = X.copy()
        df['request_dttm'] = pd.to_datetime(df['request_dttm'], format='%Y-%m-%d %H:%M:%S')
        three_months_ago = df['request_dttm'].max() - timedelta(days=90)
        filtered_data = df[df['request_dttm'] >= three_months_ago]
        filtered_data = filtered_data.set_index('id')
        df_click_per_offer  = pd.pivot_table(filtered_data, columns='id', values='cliked', fill_value=0, aggfunc=np.sum).reset_index()
        unpivot_df = pd.melt(df_click_per_offer, value_name='N_click_per_offer_sum_3m')
        unpivot_df = unpivot_df.drop(index=0).reset_index(drop=True)
        unpivot_df['udzial_klikniec_3m'] = unpivot_df['N_click_per_offer_sum_3m']/ len(filtered_data)
        
        unique_ids = unpivot_df['id'].unique()
        self._N_click_3m_mapping = dict(zip(unique_ids, unpivot_df.loc[unpivot_df['id'].isin(unique_ids), 'N_click_per_offer_sum_3m']))
        self._udzial_3m_mapping = dict(zip(unique_ids, unpivot_df.loc[unpivot_df['id'].isin(unique_ids), 'udzial_klikniec_3m']))
        return self
    
    def transform(self, X):
        df = X.copy()
        #df['N_click_per_offer_sum_3m'] = df['id'].map(self._N_click_3m_mapping)
        df['udzial_klikniec_3m'] = df['id'].map(self._udzial_3m_mapping)
        return df
    
class SixMonthsAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._N_click_6m_mapping = dict()
        self._udzial_6m_mapping = dict()
    
    def fit(self, X , y=None):
        df = X.copy()
        df['request_dttm'] = pd.to_datetime(df['request_dttm'], format='%Y-%m-%d %H:%M:%S')
        three_months_ago = df['request_dttm'].max() - timedelta(days=180)
        filtered_data = df[df['request_dttm'] >= three_months_ago]
        filtered_data = filtered_data.set_index('id')
        df_click_per_offer  = pd.pivot_table(filtered_data, columns='id', values='cliked', fill_value=0, aggfunc=np.sum).reset_index()
        unpivot_df = pd.melt(df_click_per_offer, value_name='N_click_per_offer_sum_6m')
        unpivot_df = unpivot_df.drop(index=0).reset_index(drop=True)
        unpivot_df['udzial_klikniec_6m'] = unpivot_df['N_click_per_offer_sum_6m']/ len(filtered_data)
        
        unique_ids = unpivot_df['id'].unique()
        self._N_click_6m_mapping = dict(zip(unique_ids, unpivot_df.loc[unpivot_df['id'].isin(unique_ids), 'N_click_per_offer_sum_6m']))
        self._udzial_6m_mapping = dict(zip(unique_ids, unpivot_df.loc[unpivot_df['id'].isin(unique_ids), 'udzial_klikniec_6m']))
        return self
    
    def transform(self, X):
        df = X.copy()
        #df['N_click_per_offer_sum_6m'] = df['id'].map(self._N_click_6m_mapping)
        df['udzial_klikniec_6m'] = df['id'].map(self._udzial_6m_mapping)
        return df