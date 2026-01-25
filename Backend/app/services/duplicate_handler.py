"""
DuplicateHandler: Detects and handles duplicate rows and columns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class DuplicateHandler:
    """
    Handles duplicate row and column detection and removal.
    """
    
    def __init__(self):
        pass
    
    def detect_duplicates(self, df: pd.DataFrame) -> Dict:
        """
        Detect duplicate rows and columns.
        Returns comprehensive duplicate statistics.
        """
        # Duplicate rows
        duplicate_rows = df.duplicated(keep=False)
        duplicate_row_indices = df[duplicate_rows].index.tolist()
        duplicate_row_count = duplicate_rows.sum()
        
        # Unique duplicate groups
        duplicate_groups = df[duplicate_rows].groupby(list(df.columns)).groups
        unique_duplicate_count = len(duplicate_groups)
        
        # Duplicate columns
        duplicate_columns = []
        columns = df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if df[col1].equals(df[col2]):
                    duplicate_columns.append({
                        'column1': col1,
                        'column2': col2,
                        'identical': True
                    })
        
        # Preview of duplicate rows (first 10)
        duplicate_preview = None
        if duplicate_row_count > 0:
            duplicate_preview_df = df[duplicate_rows].head(10)
            duplicate_preview = {
                'columns': duplicate_preview_df.columns.tolist(),
                'rows': duplicate_preview_df.values.tolist(),
                'indices': duplicate_preview_df.index.tolist()
            }
        
        return {
            'duplicate_row_count': int(duplicate_row_count),
            'unique_duplicate_groups': int(unique_duplicate_count),
            'duplicate_row_indices': duplicate_row_indices[:100],  # Limit to 100
            'duplicate_column_count': len(duplicate_columns),
            'duplicate_columns': duplicate_columns,
            'preview': duplicate_preview
        }
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        keep: str = 'first',
        subset: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove duplicate rows.
        
        Args:
            df: Input dataframe
            keep: 'first', 'last', or False (remove all duplicates)
            subset: Columns to consider for duplicates (None = all columns)
        
        Returns:
            Tuple of (processed_df, metadata)
        """
        rows_before = len(df)
        
        # Remove duplicates
        df_processed = df.drop_duplicates(keep=keep, subset=subset, ignore_index=False)
        
        rows_after = len(df_processed)
        rows_removed = rows_before - rows_after
        
        metadata = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_removed,
            'keep': keep,
            'subset': subset if subset else 'all_columns'
        }
        
        return df_processed, metadata
