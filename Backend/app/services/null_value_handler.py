"""
NullValueHandler: Comprehensive null value detection and handling.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class NullValueHandler:
    """
    Handles null value detection and imputation.
    Supports multiple null representations and various imputation strategies.
    """
    
    # Null value representations to detect
    NULL_VALUES = [
        None, np.nan, pd.NA,
        '', ' ', '  ',  # Empty strings
        'NaN', 'nan', 'NAN',
        'None', 'none', 'NONE',
        'Na', 'NA', 'n/a', 'N/A',
        'null', 'NULL', 'Null'
    ]
    
    def __init__(self):
        pass
    
    def detect_null_values(self, df: pd.DataFrame) -> Dict:
        """
        Detect null values across the dataset.
        Returns comprehensive null statistics.
        """
        # Convert various null representations to NaN
        df_normalized = self._normalize_nulls(df.copy())
        
        # Calculate statistics
        total_cells = df_normalized.size
        total_nulls = df_normalized.isna().sum().sum()
        null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
        
        # Per-column statistics
        column_stats = []
        for col in df_normalized.columns:
            null_count = df_normalized[col].isna().sum()
            null_pct = (null_count / len(df_normalized) * 100) if len(df_normalized) > 0 else 0
            
            column_stats.append({
                'column': col,
                'data_type': str(df_normalized[col].dtype),
                'null_count': int(null_count),
                'null_percentage': round(null_pct, 2),
                'non_null_count': int(len(df_normalized) - null_count)
            })
        
        # Sort by null percentage descending
        column_stats.sort(key=lambda x: x['null_percentage'], reverse=True)
        
        return {
            'total_null_count': int(total_nulls),
            'total_null_percentage': round(null_percentage, 2),
            'total_cells': int(total_cells),
            'columns': column_stats
        }
    
    def _normalize_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert various null representations to pandas NaN."""
        df_normalized = df.copy()
        
        # Replace string representations of null
        for null_val in self.NULL_VALUES:
            if isinstance(null_val, str):
                # Replace exact matches
                df_normalized = df_normalized.replace(null_val, np.nan)
                # Replace after stripping whitespace
                for col in df_normalized.select_dtypes(include=['object']).columns:
                    df_normalized[col] = df_normalized[col].apply(
                        lambda x: np.nan if isinstance(x, str) and x.strip() == null_val.strip() else x
                    )
        
        return df_normalized
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        strategy: str = 'mean',
        constant_value: Optional[float] = None,
        constant_string: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values according to strategy.
        
        Args:
            df: Input dataframe
            columns: List of columns to process (None = all columns)
            strategy: 'drop', 'mean', 'median', 'mode', 'constant_num', 'constant_cat'
            constant_value: Constant value for numerical columns
            constant_string: Constant value for categorical columns
        
        Returns:
            Tuple of (processed_df, metadata)
        """
        df_processed = self._normalize_nulls(df.copy())
        
        if columns is None:
            columns = list(df_processed.columns)
        
        metadata = {
            'strategy': strategy,
            'columns_processed': columns,
            'rows_before': len(df_processed),
            'rows_after': len(df_processed),
            'nulls_filled': {}
        }
        
        if strategy == 'drop':
            # Drop rows with missing values in selected columns
            rows_before = len(df_processed)
            df_processed = df_processed.dropna(subset=columns)
            rows_after = len(df_processed)
            metadata['rows_removed'] = rows_before - rows_after
            metadata['rows_after'] = rows_after
        
        else:
            # Fill missing values
            for col in columns:
                if col not in df_processed.columns:
                    continue
                
                null_count = df_processed[col].isna().sum()
                if null_count == 0:
                    continue
                
                is_numeric = pd.api.types.is_numeric_dtype(df_processed[col])
                
                if strategy == 'mean' and is_numeric:
                    fill_value = df_processed[col].mean()
                    df_processed[col].fillna(fill_value, inplace=True)
                    metadata['nulls_filled'][col] = {
                        'count': int(null_count),
                        'method': 'mean',
                        'value': float(fill_value) if not pd.isna(fill_value) else None
                    }
                
                elif strategy == 'median' and is_numeric:
                    fill_value = df_processed[col].median()
                    df_processed[col].fillna(fill_value, inplace=True)
                    metadata['nulls_filled'][col] = {
                        'count': int(null_count),
                        'method': 'median',
                        'value': float(fill_value) if not pd.isna(fill_value) else None
                    }
                
                elif strategy == 'mode':
                    mode_values = df_processed[col].mode()
                    if len(mode_values) > 0:
                        fill_value = mode_values.iloc[0]
                        df_processed[col].fillna(fill_value, inplace=True)
                        metadata['nulls_filled'][col] = {
                            'count': int(null_count),
                            'method': 'mode',
                            'value': str(fill_value) if not pd.isna(fill_value) else None
                        }
                    else:
                        # No mode available, use first non-null value or constant
                        if constant_string:
                            df_processed[col].fillna(constant_string, inplace=True)
                            metadata['nulls_filled'][col] = {
                                'count': int(null_count),
                                'method': 'constant',
                                'value': constant_string
                            }
                
                elif strategy == 'constant_num' and is_numeric:
                    if constant_value is not None:
                        df_processed[col].fillna(constant_value, inplace=True)
                        metadata['nulls_filled'][col] = {
                            'count': int(null_count),
                            'method': 'constant',
                            'value': float(constant_value)
                        }
                
                elif strategy == 'constant_cat' and not is_numeric:
                    if constant_string is not None:
                        df_processed[col].fillna(constant_string, inplace=True)
                        metadata['nulls_filled'][col] = {
                            'count': int(null_count),
                            'method': 'constant',
                            'value': constant_string
                        }
        
        return df_processed, metadata
