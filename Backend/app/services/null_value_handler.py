"""
NullValueHandler: Smart, type-aware missing value detection and imputation.

Grandmaster improvements:
  1. Detects ALL null representations (string "NaN", "None", "N/A", empty, etc.)
     without doing a full dataframe copy for detection.
  2. Imputation strategies are TYPE-SAFE:
     - 'mean' / 'median' only on genuinely numeric columns
     - 'mode' works on any dtype
     - 'ffill' / 'bfill' for time-series style data
     - 'knn' for sophisticated imputation (falls back to median if scikit not available)
     - 'constant_num' / 'constant_cat' for explicit user values
  3. ColumnRegistry-aware: if a column is label_encoded, imputation uses the
     most frequent encoded integer (mode), never mean/median on encoded values.
  4. After imputation, reports the exact fill values used per column.
  5. 'drop' strategy supports min_valid_ratio: only drop rows with < N% non-null
     values across the selected columns (smarter than drop-any).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# All string representations that mean "null"
_STR_NULLS = frozenset({
    '', ' ', '  ', 'nan', 'NaN', 'NAN',
    'none', 'None', 'NONE',
    'na', 'Na', 'NA',
    'n/a', 'N/A', 'N/a',
    'null', 'Null', 'NULL',
    'undefined', 'Undefined', 'UNDEFINED',
    'missing', 'Missing',
    '-', '--', '---',
    '?', '??',
})


class NullValueHandler:
    """
    Handles null value detection and imputation with type-awareness.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # Detection
    # ═══════════════════════════════════════════════════════════════════════

    def detect_null_values(self, df: pd.DataFrame) -> Dict:
        """
        Detect ALL null values (numeric NaN + string representations).
        Returns per-column stats without copying the entire DataFrame.
        """
        n_rows = len(df)
        total_cells = df.size

        col_null_counts = {}
        for col in df.columns:
            if df[col].dtype == object:
                is_null = df[col].isin(_STR_NULLS) | df[col].isna()
            else:
                is_null = df[col].isna()
            col_null_counts[col] = int(is_null.sum())

        total_nulls = sum(col_null_counts.values())
        null_pct_total = (total_nulls / total_cells * 100) if total_cells > 0 else 0.0

        column_stats = []
        for col in df.columns:
            null_count = col_null_counts[col]
            null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0.0
            column_stats.append({
                'column': col,
                'data_type': str(df[col].dtype),
                'null_count': null_count,
                'null_percentage': round(null_pct, 2),
                'non_null_count': n_rows - null_count,
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
            })

        column_stats.sort(key=lambda x: x['null_percentage'], reverse=True)

        return {
            'total_null_count': total_nulls,
            'total_null_percentage': round(null_pct_total, 2),
            'total_cells': total_cells,
            'total_rows': n_rows,
            'columns_with_nulls': sum(1 for v in col_null_counts.values() if v > 0),
            'columns': column_stats,
        }

    def _normalize_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise all string-null representations → np.nan in object columns only.
        Numeric columns already use NaN natively; touching them would be wasteful.
        """
        obj_cols = df.select_dtypes(include='object').columns.tolist()
        if obj_cols:
            df[obj_cols] = df[obj_cols].replace(list(_STR_NULLS), np.nan)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Imputation
    # ═══════════════════════════════════════════════════════════════════════

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        strategy: str = 'mean',
        constant_value: Optional[float] = None,
        constant_string: Optional[str] = None,
        min_valid_ratio: float = 0.0,   # for 'drop': fraction of non-null cols required
        registry=None,                   # ColumnRegistry (optional)
        session_id: str = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values.

        Strategies:
          'drop'           – drop rows containing nulls in selected columns
          'mean'           – fill with column mean  (numeric only)
          'median'         – fill with column median (numeric only)
          'mode'           – fill with most frequent value (any dtype)
          'ffill'          – forward fill (time series)
          'bfill'          – backward fill (time series)
          'knn'            – KNN imputation (numeric only; falls back to median)
          'constant_num'   – fill numeric nulls with constant_value
          'constant_cat'   – fill categorical nulls with constant_string
          'smart'          – auto-select per column: median for numeric, mode for cat

        Returns:
            (processed_df, metadata)
        """
        df_proc = self._normalize_nulls(df.copy())

        if columns is None:
            columns = list(df_proc.columns)

        # Remove columns that don't exist
        columns = [c for c in columns if c in df_proc.columns]

        metadata = {
            'strategy': strategy,
            'columns_processed': [],
            'columns_skipped': [],
            'rows_before': len(df_proc),
            'rows_after': len(df_proc),
            'nulls_filled': {},
            'warnings': [],
        }

        # ── Drop strategy ──────────────────────────────────────────────────
        if strategy == 'drop':
            rows_before = len(df_proc)
            if min_valid_ratio > 0:
                # Drop rows where the fraction of non-null values in 'columns'
                # is below the threshold
                sub = df_proc[columns]
                valid_ratio = sub.notna().mean(axis=1)
                df_proc = df_proc[valid_ratio >= min_valid_ratio]
            else:
                df_proc = df_proc.dropna(subset=columns)
            rows_after = len(df_proc)
            metadata['rows_after'] = rows_after
            metadata['rows_removed'] = rows_before - rows_after
            metadata['columns_processed'] = columns
            return df_proc, metadata

        # ── KNN strategy ───────────────────────────────────────────────────
        if strategy == 'knn':
            return self._knn_impute(df_proc, columns, metadata)

        # ── Per-column fill strategies ─────────────────────────────────────
        for col in columns:
            null_count = df_proc[col].isna().sum()

            if null_count == 0:
                metadata['columns_skipped'].append({'column': col, 'reason': 'no_nulls'})
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df_proc[col])

            # ── Registry-aware imputation override ────────────────────────
            # If a column is label/ordinal encoded, its values are integers 0,1,2...
            # Mean/median of these integers is mathematically wrong — use mode instead.
            effective_strategy = strategy
            if registry and session_id:
                role = registry.get_role(session_id, col)
                if role in {'label_encoded', 'ordinal_encoded', 'binary_encoded', 'one_hot'}:
                    if effective_strategy in {'mean', 'median'}:
                        effective_strategy = 'mode'
                        metadata['warnings'].append(
                            f"Column '{col}' is {role} — switched imputation strategy "
                            f"from '{strategy}' to 'mode' (mean/median on integer codes "
                            f"is semantically wrong)."
                        )

            # ── Apply strategy ─────────────────────────────────────────────
            fill_info = self._apply_strategy(
                df_proc, col, effective_strategy, is_numeric,
                constant_value, constant_string, metadata
            )

            if fill_info is not None:
                metadata['nulls_filled'][col] = fill_info
                metadata['columns_processed'].append(col)
            else:
                metadata['columns_skipped'].append({'column': col, 'reason': 'strategy_not_applicable'})

        return df_proc, metadata

    def _apply_strategy(
        self,
        df: pd.DataFrame,
        col: str,
        strategy: str,
        is_numeric: bool,
        constant_value,
        constant_string,
        metadata: Dict,
    ) -> Optional[Dict]:
        """Apply a single imputation strategy to one column. Returns fill info or None."""
        null_count = int(df[col].isna().sum())

        if strategy in ('mean', 'smart') and is_numeric:
            fill_val = float(df[col].mean())
            if pd.isna(fill_val):
                metadata['warnings'].append(
                    f"Column '{col}': mean is NaN (all values null). Skipped."
                )
                return None
            df[col] = df[col].fillna(fill_val)
            return {'count': null_count, 'method': 'mean', 'value': fill_val}

        elif strategy == 'median' and is_numeric:
            fill_val = float(df[col].median())
            if pd.isna(fill_val):
                metadata['warnings'].append(
                    f"Column '{col}': median is NaN (all values null). Skipped."
                )
                return None
            df[col] = df[col].fillna(fill_val)
            return {'count': null_count, 'method': 'median', 'value': fill_val}

        elif strategy in ('mode', 'smart'):
            mode_vals = df[col].mode()
            if len(mode_vals) == 0:
                metadata['warnings'].append(
                    f"Column '{col}': no mode found (all values null). Skipped."
                )
                return None
            fill_val = mode_vals.iloc[0]
            df[col] = df[col].fillna(fill_val)
            return {
                'count': null_count,
                'method': 'mode',
                'value': str(fill_val) if not pd.isna(fill_val) else None,
            }

        elif strategy == 'ffill':
            df[col] = df[col].ffill()
            remaining = int(df[col].isna().sum())
            if remaining > 0:
                # Head-of-series has no previous value; fill with bfill
                df[col] = df[col].bfill()
            return {'count': null_count, 'method': 'ffill + bfill_fallback', 'value': None}

        elif strategy == 'bfill':
            df[col] = df[col].bfill()
            remaining = int(df[col].isna().sum())
            if remaining > 0:
                df[col] = df[col].ffill()
            return {'count': null_count, 'method': 'bfill + ffill_fallback', 'value': None}

        elif strategy == 'constant_num' and is_numeric:
            if constant_value is None:
                metadata['warnings'].append(
                    f"Column '{col}': 'constant_num' strategy requires constant_value. Skipped."
                )
                return None
            df[col] = df[col].fillna(float(constant_value))
            return {'count': null_count, 'method': 'constant', 'value': float(constant_value)}

        elif strategy == 'constant_cat' and not is_numeric:
            if constant_string is None:
                metadata['warnings'].append(
                    f"Column '{col}': 'constant_cat' strategy requires constant_string. Skipped."
                )
                return None
            df[col] = df[col].fillna(constant_string)
            return {'count': null_count, 'method': 'constant', 'value': constant_string}

        elif strategy in ('mean', 'median') and not is_numeric:
            # Graceful fallback: use mode for categorical
            return self._apply_strategy(df, col, 'mode', is_numeric,
                                        constant_value, constant_string, metadata)

        return None

    def _knn_impute(
        self,
        df: pd.DataFrame,
        columns: List[str],
        metadata: Dict,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        KNN imputation using sklearn.impute.KNNImputer.
        Only operates on numeric columns (KNN requires numeric data).
        Falls back to median for non-numeric columns.
        """
        try:
            from sklearn.impute import KNNImputer
            numeric_cols = [c for c in columns
                            if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()]
            cat_cols = [c for c in columns
                        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()]

            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                for col in numeric_cols:
                    metadata['nulls_filled'][col] = {
                        'count': '?',   # already filled
                        'method': 'knn_5neighbors',
                        'value': None
                    }
                metadata['columns_processed'].extend(numeric_cols)

            # Fallback to mode for categorical
            for col in cat_cols:
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    fill_val = mode_vals.iloc[0]
                    null_count = int(df[col].isna().sum())
                    df[col] = df[col].fillna(fill_val)
                    metadata['nulls_filled'][col] = {
                        'count': null_count, 'method': 'mode_fallback_for_knn',
                        'value': str(fill_val)
                    }
                    metadata['columns_processed'].append(col)

        except ImportError:
            metadata['warnings'].append(
                'scikit-learn KNNImputer not found. Falling back to median imputation.'
            )
            # Fallback
            for col in columns:
                if df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_val = float(df[col].median())
                        null_count = int(df[col].isna().sum())
                        if not pd.isna(fill_val):
                            df[col] = df[col].fillna(fill_val)
                            metadata['nulls_filled'][col] = {
                                'count': null_count, 'method': 'median_knn_fallback',
                                'value': fill_val
                            }
                            metadata['columns_processed'].append(col)

        metadata['rows_after'] = len(df)
        return df, metadata
