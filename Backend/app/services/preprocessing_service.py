import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict
from .data_service import DataService

class PreprocessingService:
    """
    Service to preprocess datasets: handle missing values, encode categoricals, scale features, split data.
    """
    def __init__(self):
        self.data_service = DataService()

    def preprocess_data(self, session_id: str, config: Dict) -> Dict:
        """
        Preprocess the dataset according to configuration.
        Returns processed data splits.
        """
        df = self.data_service.get_data(session_id)

        # Handle missing values
        df_processed = self._handle_missing_values(df, config['missing_strategy'])

        # Encode categorical features
        df_processed, encoders = self._encode_categorical(df_processed)

        # Split features and target (will be done during training, but prepare here)
        # For now, just prepare the data

        # Feature scaling if requested
        if config['scaling']:
            df_processed, scaler = self._scale_features(df_processed)
        else:
            scaler = None

        # Store processed data
        processed_data = {
            'data': df_processed,
            'encoders': encoders,
            'scaler': scaler,
            'original_columns': list(df.columns)
        }

        self.data_service.store_processed_data(session_id, processed_data)

        return {
            "message": "Data preprocessed successfully",
            "shape": df_processed.shape,
            "columns": list(df_processed.columns)
        }

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        df = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if strategy == 'drop':
                    df = df.dropna(subset=[col])
                elif strategy == 'mean' and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> tuple:
        """Encode categorical features using Label Encoding"""
        df = df.copy()
        encoders = {}

        for col in df.select_dtypes(include=['object', 'category']):
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder

        return df, encoders

    def _scale_features(self, df: pd.DataFrame) -> tuple:
        """Scale numerical features using StandardScaler"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df, scaler