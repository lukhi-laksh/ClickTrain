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
        Detect duplicate rows.

        rows_to_remove  – rows that will actually be DELETED when
                          drop_duplicates(keep='first') is called.
                          Formula: df.duplicated(keep='first').sum()
        preview         – ONLY the rows that will be removed (not the kept copy).
        groups          – each distinct repeated row pattern, sorted by frequency
                          descending, limited to top 15.  Used by the UI to show
                          a compact frequency summary.
        """
        MAX_PREVIEW = 20
        total_rows  = len(df)

        # keep=False → marks every member of any duplicate group
        all_dup_mask        = df.duplicated(keep=False)
        duplicate_row_count = int(all_dup_mask.sum())

        # keep='first' → marks only the EXTRA copies that will be deleted
        remove_mask    = df.duplicated(keep='first')
        rows_to_remove = int(remove_mask.sum())

        unique_duplicate_groups = duplicate_row_count - rows_to_remove

        # ── Preview: rows that will be REMOVED (not the kept originals) ──────
        duplicate_preview = None
        if rows_to_remove > 0:
            preview_df = df[remove_mask].head(MAX_PREVIEW)
            duplicate_preview = {
                'columns':         preview_df.columns.tolist(),
                'rows': [
                    [None if (isinstance(v, float) and __import__('math').isnan(v)) else v
                     for v in row]
                    for row in preview_df.values.tolist()
                ],
                'indices':         preview_df.index.tolist(),
                'total_to_remove': rows_to_remove,
                'preview_capped':  rows_to_remove > MAX_PREVIEW,
            }

        # ── Group frequency (top 15 patterns, most repeated first) ───────────
        groups = []
        if rows_to_remove > 0:
            try:
                # Convert to str for NaN-safe groupby
                df_str = df.astype(str)
                col_list = df_str.columns.tolist()
                vc = df_str.groupby(col_list, sort=False).size()
                dup_vc = vc[vc > 1].sort_values(ascending=False).head(15)
                for idx, cnt in dup_vc.items():
                    vals = list(idx) if isinstance(idx, tuple) else [idx]
                    groups.append({
                        'values':    vals,        # string representation of each cell
                        'count':     int(cnt),    # total appearances in dataset
                        'to_remove': int(cnt) - 1 # copies that will be deleted
                    })
            except Exception:
                pass   # best-effort — never crash on group analysis

        return {
            # IMPORTANT: duplicate_row_count is set equal to rows_to_remove
            # so that any consumer (including old cached JS) always gets the
            # correct user-facing count: rows that will actually be DELETED.
            'duplicate_row_count':     rows_to_remove,   # = keep='first' count
            'rows_to_remove':          rows_to_remove,   # same value, explicit field
            'unique_duplicate_groups': unique_duplicate_groups,
            'duplicate_row_indices':   df[remove_mask].index.tolist()[:100],
            'duplicate_column_count':  0,
            'duplicate_columns':       [],
            'total_rows':              total_rows,
            'unique_rows':             total_rows - rows_to_remove,   # rows remaining after dedup
            'preview':                 duplicate_preview,
            'groups':                  groups,
            'columns':                 df.columns.tolist(),
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
