"""
OutlierHandler: Detects and handles outliers using IQR and Z-score methods.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class OutlierHandler:
    """
    Handles outlier detection and treatment using IQR and Z-score methods.
    """
    
    def __init__(self):
        pass
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Dict:
        """
        Detect outliers in specified columns.
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: For z-score, number of standard deviations (default 3.0)
        
        Returns:
            Dictionary with outlier statistics
        """
        outlier_info = {
            'method': method,
            'threshold': threshold,
            'columns': {},
            'total_outlier_rows': 0,
            'outlier_row_indices': set()
        }
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if method == 'iqr':
                outliers, stats = self._detect_iqr_outliers(df[col])
            elif method == 'zscore':
                outliers, stats = self._detect_zscore_outliers(df[col], threshold)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outlier_indices = df[df[col].index.isin(outliers)].index.tolist()
            outlier_info['outlier_row_indices'].update(outlier_indices)
            
            outlier_info['columns'][col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df) * 100) if len(df) > 0 else 0,
                'outlier_indices': outlier_indices[:50],  # Limit to 50
                'stats': stats
            }
        
        # Get unique outlier rows
        outlier_info['total_outlier_rows'] = len(outlier_info['outlier_row_indices'])
        outlier_info['outlier_percentage'] = (
            outlier_info['total_outlier_rows'] / len(df) * 100
        ) if len(df) > 0 else 0
        outlier_info['outlier_row_indices'] = list(outlier_info['outlier_row_indices'])[:100]
        
        return outlier_info
    
    def _detect_iqr_outliers(self, series: pd.Series) -> Tuple[List, Dict]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        
        stats = {
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'formula': 'Outliers: values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR'
        }
        
        return outliers, stats
    
    def _detect_zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> Tuple[List, Dict]:
        """Detect outliers using Z-score method."""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return [], {
                'mean': float(mean),
                'std': 0.0,
                'threshold': threshold,
                'formula': 'Z-score = (x - μ) / σ'
            }
        
        z_scores = np.abs((series - mean) / std)
        outliers = series[z_scores > threshold].index.tolist()
        
        stats = {
            'mean': float(mean),
            'std': float(std),
            'threshold': threshold,
            'formula': f'Outliers: |z| > {threshold} where z = (x - μ) / σ'
        }
        
        return outliers, stats
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        action: str = 'remove',
        threshold: float = 3.0,
        cap_percentile: float = 0.05
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle outliers according to specified action.
        
        Args:
            action: 'remove', 'cap', or 'flag'
            cap_percentile: Percentile for capping (0.05 = cap at 5th and 95th percentiles)
        
        Returns:
            Tuple of (processed_df, metadata)
        """
        df_processed = df.copy()
        metadata = {
            'method': method,
            'action': action,
            'columns': {},
            'rows_before': len(df_processed),
            'rows_after': len(df_processed),
            'rows_removed': 0
        }
        
        outlier_indices = set()
        
        for col in columns:
            if col not in df_processed.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                continue
            
            # Detect outliers
            if method == 'iqr':
                outliers, stats = self._detect_iqr_outliers(df_processed[col])
            elif method == 'zscore':
                outliers, stats = self._detect_zscore_outliers(df_processed[col], threshold)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            col_outlier_indices = df_processed[df_processed[col].index.isin(outliers)].index.tolist()
            outlier_indices.update(col_outlier_indices)
            
            if action == 'remove':
                # Will be handled after loop
                pass
            
            elif action == 'cap':
                # Winsorization: cap at percentiles
                lower_percentile = df_processed[col].quantile(cap_percentile)
                upper_percentile = df_processed[col].quantile(1 - cap_percentile)
                
                df_processed[col] = df_processed[col].clip(
                    lower=lower_percentile,
                    upper=upper_percentile
                )
                
                metadata['columns'][col] = {
                    'outliers_capped': len(outliers),
                    'lower_cap': float(lower_percentile),
                    'upper_cap': float(upper_percentile),
                    'stats': stats
                }
            
            elif action == 'flag':
                # Add outlier flag column
                flag_col = f'{col}_outlier'
                df_processed[flag_col] = df_processed.index.isin(outliers).astype(int)
                
                metadata['columns'][col] = {
                    'outliers_flagged': len(outliers),
                    'flag_column': flag_col,
                    'stats': stats
                }
        
        # Remove outlier rows if action is 'remove'
        if action == 'remove':
            rows_before = len(df_processed)
            df_processed = df_processed.drop(index=list(outlier_indices))
            rows_after = len(df_processed)
            
            metadata['rows_removed'] = rows_before - rows_after
            metadata['rows_after'] = rows_after
            
            for col in columns:
                if col in df_processed.columns:
                    col_outlier_count = len([
                        idx for idx in outlier_indices
                        if idx in df_processed.index
                    ])
                    metadata['columns'][col] = {
                        'outliers_removed': col_outlier_count
                    }
        
        return df_processed, metadata
