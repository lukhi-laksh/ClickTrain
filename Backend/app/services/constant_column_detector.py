"""
ConstantColumnDetector: Detects constant and low-variance columns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class ConstantColumnDetector:
    """
    Detects constant columns (zero variance) and optionally low-variance columns.
    """
    
    def __init__(self, variance_threshold: float = 0.0):
        """
        Args:
            variance_threshold: Minimum variance threshold (0.0 = only constant columns)
        """
        self.variance_threshold = variance_threshold
    
    def detect_constant_columns(self, df: pd.DataFrame) -> Dict:
        """
        Detect constant columns (zero variance).
        Returns list of constant columns with their constant values.
        """
        constant_columns = []
        
        for col in df.columns:
            # Check if column has constant values
            unique_values = df[col].nunique()
            
            if unique_values <= 1:
                # Constant column
                constant_value = df[col].iloc[0] if len(df) > 0 else None
                constant_columns.append({
                    'column': col,
                    'constant_value': str(constant_value) if constant_value is not None else 'None',
                    'unique_count': int(unique_values),
                    'data_type': str(df[col].dtype)
                })
            
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check variance for numerical columns
                variance = df[col].var()
                if pd.isna(variance) or variance <= self.variance_threshold:
                    constant_value = df[col].iloc[0] if len(df) > 0 else None
                    constant_columns.append({
                        'column': col,
                        'constant_value': str(constant_value) if constant_value is not None else 'None',
                        'unique_count': int(unique_values),
                        'variance': float(variance) if not pd.isna(variance) else 0.0,
                        'data_type': str(df[col].dtype)
                    })
        
        return {
            'constant_column_count': len(constant_columns),
            'constant_columns': constant_columns
        }
    
    def remove_columns(self, df: pd.DataFrame, columns: List[str]) -> tuple:
        """
        Remove specified columns from dataframe.
        
        Returns:
            Tuple of (processed_df, metadata)
        """
        rows_before = len(df)
        cols_before = len(df.columns)
        
        # Remove columns
        columns_to_remove = [col for col in columns if col in df.columns]
        df_processed = df.drop(columns=columns_to_remove)
        
        cols_after = len(df_processed.columns)
        cols_removed = cols_before - cols_after
        
        metadata = {
            'columns_removed': columns_to_remove,
            'columns_before': cols_before,
            'columns_after': cols_after,
            'columns_removed_count': cols_removed,
            'rows_before': rows_before,
            'rows_after': len(df_processed)
        }
        
        return df_processed, metadata
