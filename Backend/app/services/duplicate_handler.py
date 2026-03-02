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
        Detect duplicate rows. Fast path: uses pandas vectorized .duplicated().
        """
        total_rows = len(df)
        # Boolean mask of ALL duplicate rows (including originals)
        dup_mask = df.duplicated(keep=False)
        duplicate_row_count = int(dup_mask.sum())

        # Number of unique duplicate groups = total duplicated rows - rows that would be kept
        # i.e. dup_mask.sum() - duplicated(keep='first').sum()
        unique_duplicate_count = int(duplicate_row_count - df.duplicated(keep='first').sum()) if duplicate_row_count else 0

        # Preview: first 10 duplicated rows only (no full groupby)
        duplicate_preview = None
        if duplicate_row_count > 0:
            preview_df = df[dup_mask].head(10)
            duplicate_preview = {
                'columns': preview_df.columns.tolist(),
                'rows':    preview_df.values.tolist(),
                'indices': preview_df.index.tolist()
            }

        return {
            'duplicate_row_count':   duplicate_row_count,
            'unique_duplicate_groups': unique_duplicate_count,
            'duplicate_row_indices': df[dup_mask].index.tolist()[:100],
            'duplicate_column_count': 0,   # not computed (unused by frontend)
            'duplicate_columns': [],
            'total_rows': total_rows,
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
