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
        FAST PATH: no full DataFrame copy — uses vectorized replace in-place on a
        column-by-column basis, skipping numeric columns that can't hold string nulls.
        """
        # Build the normalized null mask without copying the whole dataframe.
        # For numeric cols, pandas isnull already covers NaN/None/pd.NA.
        # For object cols, we also need to catch string-null representations.
        str_nulls = self._str_null_set()
        null_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        for col in df.columns:
            if df[col].dtype == object:
                null_mask[col] = df[col].isin(str_nulls) | df[col].isna()
            else:
                null_mask[col] = df[col].isna()

        # Aggregate statistics from the mask (no copy of data values needed)
        col_null_counts = null_mask.sum()
        total_nulls = int(col_null_counts.sum())
        total_cells = df.size
        null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
        n = len(df)

        column_stats = []
        for col in df.columns:
            null_count = int(col_null_counts[col])
            null_pct = (null_count / n * 100) if n > 0 else 0
            column_stats.append({
                'column': col,
                'data_type': str(df[col].dtype),
                'null_count': null_count,
                'null_percentage': round(null_pct, 2),
                'non_null_count': n - null_count
            })

        column_stats.sort(key=lambda x: x['null_percentage'], reverse=True)
        return {
            'total_null_count': total_nulls,
            'total_null_percentage': round(null_percentage, 2),
            'total_cells': total_cells,
            'columns': column_stats
        }
    
    def _str_null_set(self):
        """Return a frozenset of string representations that mean 'null'."""
        str_nulls = [v for v in self.NULL_VALUES if isinstance(v, str)]
        stripped = {v.strip() for v in str_nulls}
        return frozenset(str_nulls) | stripped

    def _normalize_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert various null representations to pandas NaN. Fully vectorized.
        Only operates on object-dtype columns to avoid touching numeric cols."""
        all_str_nulls = list(self._str_null_set())
        # Only replace on object columns — numeric cols cannot hold string nulls
        obj_cols = df.select_dtypes(include='object').columns
        if len(obj_cols):
            df[obj_cols] = df[obj_cols].replace(all_str_nulls, np.nan)
        return df
    
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
        df_processed = self._normalize_nulls(df.copy())  # copy needed here — we mutate values
        
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
