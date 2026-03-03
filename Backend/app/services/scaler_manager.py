"""
ScalerManager: Safe, registry-aware feature scaling — grandmaster edition.

Key safety guarantees:
  1. Columns that are label_encoded, ordinal_encoded, one_hot, or binary_encoded
     are BLOCKED from scaling — their integer codes are meaningless after scaling
     (e.g. MinMax would squash 0/1 labels to 0.0/1.0 but then StandardScaler on
     the same column would produce ~0.0 for everything, destroying information).
  2. A column that has already been scaled once is not re-scaled.
  3. Constant columns (std = 0) are skipped and warned about (StandardScaler
     would produce NaN for them; MinMax would divide by zero).
  4. Columns with NaN values are handled gracefully — the scaler is fitted on
     non-null values and NaN positions are preserved after scaling.
  5. Returns a rich `skipped_columns` list so the frontend can inform the user
     exactly which columns were not scaled and why.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import base64


_SCALER_CLASSES = {
    'standard': StandardScaler,
    'minmax':   MinMaxScaler,
    'robust':   RobustScaler,
}

# Roles that must never be scaled
_BLOCKED_ROLES = {
    'label_encoded',
    'ordinal_encoded',
    'one_hot',
    'binary_encoded',
    'scaled',
    'dropped',
    'original_categorical',   # raw string column — scaling makes no sense
}


class ScalerManager:
    """
    Manages scaling with full role-awareness via ColumnRegistry.
    """

    def __init__(self):
        # {session_id: {column_name: fitted_scaler}}
        self.scalers: Dict[str, Dict[str, Any]] = {}

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'standard',
        session_id: str = None,
        registry=None,              # ColumnRegistry instance
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale numeric features.

        Args:
            method: 'standard' | 'minmax' | 'robust'
            registry: ColumnRegistry — when provided, only 'original_numeric'
                      and 'target_encoded' columns are allowed through.

        Returns:
            (scaled_df, metadata)
        """
        if method not in _SCALER_CLASSES:
            raise ValueError(
                f"Unknown scaling method: '{method}'. "
                f"Choose from: {list(_SCALER_CLASSES.keys())}"
            )

        df_scaled = df.copy()
        self.scalers.setdefault(session_id, {})

        scaled_cols, skipped_cols, warnings_list = [], [], []

        for col in columns:
            if col not in df_scaled.columns:
                skipped_cols.append({'column': col, 'reason': 'not_found'})
                continue

            # ── Registry guard ────────────────────────────────────────────
            if registry:
                role = registry.get_role(session_id, col)
                if role in _BLOCKED_ROLES:
                    skipped_cols.append({
                        'column': col,
                        'reason': 'blocked_by_role',
                        'role': role,
                        'explanation': _role_explanation(role),
                    })
                    warnings_list.append(
                        f"Column '{col}' (role: {role}) was skipped — "
                        f"{_role_explanation(role)}"
                    )
                    continue
            else:
                # Fallback without registry: at minimum skip non-numeric
                if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                    skipped_cols.append({'column': col, 'reason': 'not_numeric'})
                    continue

            # ── Numeric dtype guard ───────────────────────────────────────
            if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                skipped_cols.append({'column': col, 'reason': 'not_numeric_dtype'})
                continue

            # ── Constant column guard ─────────────────────────────────────
            col_std = df_scaled[col].std(skipna=True)
            if col_std == 0 or pd.isna(col_std):
                skipped_cols.append({
                    'column': col,
                    'reason': 'constant_column',
                    'explanation': 'Standard deviation is 0; scaling would produce NaN/Inf.'
                })
                warnings_list.append(
                    f"Column '{col}' is constant (std=0) — scaling skipped."
                )
                continue

            # ── NaN-safe scaling ──────────────────────────────────────────
            null_mask = df_scaled[col].isna()
            has_nulls = null_mask.any()

            if has_nulls:
                # Fit on non-null subset
                non_null_values = df_scaled.loc[~null_mask, col].values.reshape(-1, 1)
                scaler = _SCALER_CLASSES[method]()
                scaler.fit(non_null_values)
                all_values = df_scaled[col].values.reshape(-1, 1)
                # Only transform non-null positions
                scaled_vals = df_scaled[col].copy()
                scaled_vals[~null_mask] = scaler.transform(non_null_values).flatten()
                df_scaled[col] = scaled_vals
            else:
                col_data = df_scaled[col].values.reshape(-1, 1)
                scaler = _SCALER_CLASSES[method]()
                df_scaled[col] = scaler.fit_transform(col_data).flatten()

            # ── Store scaler ──────────────────────────────────────────────
            self.scalers[session_id][col] = scaler

            if registry:
                registry.mark_scaled(session_id, col)

            # ── Collect stats ─────────────────────────────────────────────
            scaled_cols.append({
                'column': col,
                'had_nulls': bool(has_nulls),
                'scaler_params': _extract_params(scaler, method),
            })

        metadata = {
            'method': method,
            'columns_scaled': [s['column'] for s in scaled_cols],
            'columns_skipped': skipped_cols,
            'scaler_details': scaled_cols,
            'warnings': warnings_list,
        }
        return df_scaled, metadata

    # ── Helpers ────────────────────────────────────────────────────────────

    def get_scalers(self, session_id: str) -> Dict:
        return self.scalers.get(session_id, {})

    def serialize_scalers(self, session_id: str) -> Dict:
        if session_id not in self.scalers:
            return {}
        serialized = {}
        for col, scaler in self.scalers[session_id].items():
            scaler_b64 = base64.b64encode(pickle.dumps(scaler)).decode('utf-8')
            serialized[col] = {
                'type': type(scaler).__name__,
                'data': scaler_b64,
                'params': _extract_params(scaler, ''),
            }
        return serialized

    def clear_session(self, session_id: str):
        self.scalers.pop(session_id, None)


# ── Module-level utilities ─────────────────────────────────────────────────

def _role_explanation(role: str) -> str:
    explanations = {
        'label_encoded':       'Scaling integer class labels destroys their categorical meaning.',
        'ordinal_encoded':     'Scaling ordinal codes compresses rank distances — use as-is.',
        'one_hot':             'One-Hot dummy columns are already 0/1 — scaling is harmful.',
        'binary_encoded':      'Binary 0/1 columns must not be scaled (MinMax → all zeros).',
        'scaled':              'Column is already scaled — re-scaling would double-transform.',
        'dropped':             'Column was removed from the dataset.',
        'original_categorical':'Column is still text — scale only after encoding.',
    }
    return explanations.get(role, 'Role is not safe to scale.')


def _extract_params(scaler, method: str) -> Dict:
    params = {}
    if hasattr(scaler, 'mean_'):
        params['mean'] = [float(v) for v in np.array(scaler.mean_).flatten()]
    if hasattr(scaler, 'scale_'):
        params['scale'] = [float(v) for v in np.array(scaler.scale_).flatten()]
    if hasattr(scaler, 'data_min_'):
        params['data_min'] = [float(v) for v in np.array(scaler.data_min_).flatten()]
        params['data_max'] = [float(v) for v in np.array(scaler.data_max_).flatten()]
    if hasattr(scaler, 'center_'):
        params['center'] = [float(v) for v in np.array(scaler.center_).flatten()]
    return params
