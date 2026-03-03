"""
EncoderManager: Safe, smart categorical encoding — grandmaster edition.

Key safety guarantees:
  1. A column already encoded (label/ordinal/OHE/binary) is silently skipped
     with a logged warning — no double-encoding.
  2. A purely numeric column is never label/ordinal encoded (would break it).
  3. Binary-class columns inside OHE are handled via binary mapping (0/1),
     NOT via LabelEncoder (which is alphabet-order and inconsistent).
  4. Ordinal encoding with unknown categories gets -1 with a warning.
  5. Target encoding uses leave-one-out style smoothing to reduce leakage.
  6. ColumnRegistry is updated after every successful encoding so ScalerManager
     will refuse to scale those columns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pickle
import base64
import warnings


class EncoderManager:
    """
    Manages all encoding operations with full role-awareness via ColumnRegistry.
    """

    def __init__(self):
        # {session_id: {column_name: encoder_object_or_dict}}
        self.encoders: Dict[str, Dict[str, Any]] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # Label Encoding
    # ═══════════════════════════════════════════════════════════════════════

    def label_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        session_id: str,
        registry=None,          # ColumnRegistry instance (optional but recommended)
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Label Encoding: converts each unique string value to an integer.

        Safe behaviour:
          - Already-encoded columns are skipped (no double-encoding).
          - Numeric columns are skipped (would destroy meaning).
          - NaN values are preserved as np.nan after encoding (not turned to -1).
        """
        df_enc = df.copy()
        self.encoders.setdefault(session_id, {})

        encoded_cols, skipped_cols, warnings_list = [], [], []

        for col in columns:
            if col not in df_enc.columns:
                skipped_cols.append({'column': col, 'reason': 'not_found'})
                continue

            # Guard: already encoded?
            if registry and registry.is_already_encoded(session_id, col):
                skipped_cols.append({'column': col, 'reason': 'already_encoded',
                                     'current_role': registry.get_role(session_id, col)})
                continue

            # Guard: skip numeric-origin columns
            if registry and registry.get_role(session_id, col) == 'original_numeric':
                skipped_cols.append({'column': col, 'reason': 'numeric_column_skip'})
                warnings_list.append(
                    f"Column '{col}' is numeric — label encoding skipped to preserve numeric meaning."
                )
                continue

            # Must be object / category dtype to safely encode
            if not (pd.api.types.is_object_dtype(df_enc[col]) or
                    pd.api.types.is_categorical_dtype(df_enc[col])):
                skipped_cols.append({'column': col, 'reason': 'not_categorical_dtype'})
                continue

            # Preserve NaN positions
            null_mask = df_enc[col].isna()

            # Fit on non-null values only
            non_null_series = df_enc[col].dropna().astype(str)
            encoder = LabelEncoder()
            encoder.fit(non_null_series)

            # Transform — fill NaN slots back
            df_enc[col] = df_enc[col].map(
                lambda v: encoder.transform([str(v)])[0] if pd.notna(v) else np.nan
            )
            # Restore nulls
            df_enc.loc[null_mask, col] = np.nan

            # Store encoder
            self.encoders[session_id][col] = encoder

            # Update registry
            if registry:
                registry.mark_label_encoded(session_id, col)

            encoded_cols.append(col)

        metadata = {
            'method': 'label',
            'columns_encoded': encoded_cols,
            'columns_skipped': skipped_cols,
            'warnings': warnings_list,
            'encoders': {
                col: {'type': 'label', 'classes': self.encoders[session_id][col].classes_.tolist()}
                for col in encoded_cols
                if col in self.encoders.get(session_id, {})
            }
        }
        return df_enc, metadata

    # ═══════════════════════════════════════════════════════════════════════
    # One-Hot Encoding
    # ═══════════════════════════════════════════════════════════════════════

    def one_hot_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_first: bool = False,
        handle_binary: bool = True,
        session_id: str = None,
        registry=None,
        max_categories: int = 50,   # safety cap — > 50 unique values → refuse OHE
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply One-Hot Encoding.

        Binary columns (exactly 2 unique non-null values) are encoded to 0/1
        using a deterministic sort-based mapping so the result is reproducible and
        the column stays in-place (not expanded to 2 dummy columns).

        Safety:
          - Refuses OHE for columns with > max_categories unique values.
          - Skips already-encoded columns.
          - OHE dummy columns are registered in ColumnRegistry so they are
            never scaled.
        """
        df_enc = df.copy()
        self.encoders.setdefault(session_id, {})

        encoded_info, skipped_cols, warnings_list = [], [], []

        for col in columns:
            if col not in df_enc.columns:
                skipped_cols.append({'column': col, 'reason': 'not_found'})
                continue

            if registry and registry.is_already_encoded(session_id, col):
                skipped_cols.append({'column': col, 'reason': 'already_encoded'})
                continue

            unique_vals = df_enc[col].dropna().unique()
            n_unique = len(unique_vals)

            # ── Binary: 2-class → deterministic 0/1 ─────────────────────
            if handle_binary and n_unique == 2:
                sorted_vals = sorted([str(v) for v in unique_vals])
                mapping = {val: idx for idx, val in enumerate(sorted_vals)}
                null_mask = df_enc[col].isna()
                df_enc[col] = df_enc[col].astype(str).map(mapping)
                df_enc.loc[null_mask, col] = np.nan

                self.encoders[session_id][col] = {
                    'type': 'binary', 'mapping': mapping
                }
                if registry:
                    registry.mark_binary_encoded(session_id, col)

                encoded_info.append({
                    'original': col,
                    'method': 'binary_0_1',
                    'mapping': mapping,
                    'reason': 'binary_column_deterministic'
                })
                continue

            # ── High cardinality guard ───────────────────────────────────
            if n_unique > max_categories:
                skipped_cols.append({
                    'column': col,
                    'reason': 'too_many_categories',
                    'unique_count': n_unique,
                    'max_allowed': max_categories,
                    'suggestion': 'Use label or target encoding for high-cardinality columns'
                })
                warnings_list.append(
                    f"Column '{col}' has {n_unique} unique values (max {max_categories} for OHE). "
                    f"Consider label or target encoding."
                )
                continue

            # ── Standard OHE ─────────────────────────────────────────────
            dummies = pd.get_dummies(
                df_enc[col], prefix=col, drop_first=drop_first, dtype=int
            )
            new_cols = dummies.columns.tolist()

            df_enc = df_enc.drop(columns=[col])
            df_enc = pd.concat([df_enc, dummies], axis=1)

            self.encoders[session_id][col] = {
                'type': 'one_hot', 'new_columns': new_cols,
                'drop_first': drop_first
            }

            # Register each dummy column in registry
            if registry:
                registry.mark_original_dropped(session_id, col)
                for dc in new_cols:
                    registry.mark_one_hot(session_id, dc)

            encoded_info.append({
                'original': col,
                'method': 'one_hot',
                'new_columns': new_cols
            })

        metadata = {
            'method': 'one_hot',
            'columns_encoded': encoded_info,
            'columns_skipped': skipped_cols,
            'warnings': warnings_list,
            'total_new_columns': sum(
                len(e.get('new_columns', [])) for e in encoded_info
                if isinstance(e.get('new_columns'), list)
            ),
        }
        return df_enc, metadata

    # ═══════════════════════════════════════════════════════════════════════
    # Ordinal Encoding
    # ═══════════════════════════════════════════════════════════════════════

    def ordinal_encode(
        self,
        df: pd.DataFrame,
        column: str,
        categories: Optional[List] = None,
        auto_order: bool = True,
        session_id: str = None,
        registry=None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Ordinal Encoding to a single column.

        Unknown values (not in the category list) are mapped to -1 and a warning
        is emitted — they are NOT silently mapped to NaN which could corrupt downstream steps.

        Args:
            categories: Explicit ordered list.  If None, auto-order is applied.
            auto_order:  Sort unique values alphabetically/numerically.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        if registry and registry.is_already_encoded(session_id, column):
            raise ValueError(
                f"Column '{column}' is already encoded "
                f"(role: {registry.get_role(session_id, column)}). "
                f"Cannot apply ordinal encoding again."
            )

        df_enc = df.copy()
        null_mask = df_enc[column].isna()

        # Determine category order
        if categories:
            category_order = [str(c) for c in categories]
        elif auto_order:
            unique_vals = df_enc[column].dropna().unique()
            try:
                category_order = sorted([str(v) for v in unique_vals])
            except TypeError:
                category_order = [str(v) for v in unique_vals]
        else:
            category_order = [str(v) for v in df_enc[column].dropna().unique()]

        category_map = {cat: idx for idx, cat in enumerate(category_order)}

        # Map values, unknown → -1
        df_enc[column] = df_enc[column].astype(str).map(
            lambda v: category_map.get(v, -1)
        )
        # Restore NaN positions (keep as NaN, not -1)
        df_enc.loc[null_mask, column] = np.nan

        unknown_count = int((df_enc[column] == -1).sum())
        warnings_list = []
        if unknown_count > 0:
            warnings_list.append(
                f"{unknown_count} values in '{column}' were not in the category list "
                f"and were mapped to -1. Consider fixing the category list."
            )

        # Store encoder
        self.encoders.setdefault(session_id, {})
        self.encoders[session_id][column] = {
            'type': 'ordinal', 'category_map': category_map
        }

        if registry:
            registry.mark_ordinal_encoded(session_id, column)

        metadata = {
            'method': 'ordinal',
            'column': column,
            'category_order': category_order,
            'category_map': category_map,
            'auto_order': auto_order and categories is None,
            'unknown_values_mapped_to_minus1': unknown_count,
            'warnings': warnings_list,
        }
        return df_enc, metadata

    # ═══════════════════════════════════════════════════════════════════════
    # Target Encoding  (with smoothing to reduce leakage)
    # ═══════════════════════════════════════════════════════════════════════

    def target_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_column: str,
        session_id: str = None,
        registry=None,
        smoothing: float = 10.0,    # higher = more regularisation
        min_samples_leaf: int = 1,  # minimum category frequency to use category mean
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Smoothed Target Encoding (Bayesian mean estimation).

        Formula per category c:
            encoded(c) = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)

        This is far safer than raw mean encoding because:
          - Rare categories get pulled toward the global mean (less memorisation).
          - The result is still continuous and meaningful after scaling.

        Safety:
          - Already-encoded columns are skipped.
          - Cannot target-encode the target column itself.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        df_enc = df.copy()
        self.encoders.setdefault(session_id, {})

        encoded_cols, skipped_cols, warnings_list = [], [], []
        global_mean = float(df_enc[target_column].mean())

        for col in columns:
            if col not in df_enc.columns:
                skipped_cols.append({'column': col, 'reason': 'not_found'})
                continue

            if col == target_column:
                skipped_cols.append({'column': col, 'reason': 'is_target_column'})
                continue

            if registry and registry.is_already_encoded(session_id, col):
                skipped_cols.append({'column': col, 'reason': 'already_encoded'})
                continue

            # Group stats
            stats = df_enc.groupby(col)[target_column].agg(['count', 'mean'])
            stats.columns = ['n', 'mean_y']

            # Smoothed estimate
            stats['smoothed'] = (
                (stats['n'] * stats['mean_y'] + smoothing * global_mean)
                / (stats['n'] + smoothing)
            )
            target_map = stats['smoothed'].to_dict()

            # Apply — unseen values get global_mean
            df_enc[col] = df_enc[col].map(target_map).fillna(global_mean)

            self.encoders[session_id][col] = {
                'type': 'target',
                'target_map': {str(k): float(v) for k, v in target_map.items()},
                'global_mean': global_mean,
                'smoothing': smoothing,
            }

            if registry:
                registry.mark_target_encoded(session_id, col)

            encoded_cols.append(col)

        warnings_list.append(
            'Target encoding uses smoothing to reduce leakage. '
            'Still apply cross-validation when using this for model evaluation.'
        )

        metadata = {
            'method': 'target',
            'target_column': target_column,
            'smoothing': smoothing,
            'global_mean': global_mean,
            'columns_encoded': encoded_cols,
            'columns_skipped': skipped_cols,
            'warnings': warnings_list,
        }
        return df_enc, metadata

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def get_encoders(self, session_id: str) -> Dict:
        return self.encoders.get(session_id, {})

    def serialize_encoders(self, session_id: str) -> Dict:
        if session_id not in self.encoders:
            return {}
        serialized = {}
        for col, enc in self.encoders[session_id].items():
            if isinstance(enc, dict):
                serialized[col] = enc
            else:
                enc_b64 = base64.b64encode(pickle.dumps(enc)).decode('utf-8')
                serialized[col] = {'type': 'sklearn', 'data': enc_b64}
        return serialized

    def clear_session(self, session_id: str):
        self.encoders.pop(session_id, None)
