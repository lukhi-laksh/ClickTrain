"""
ColumnRegistry: Tracks the role and state of every column per session.

This is the BRAIN of safe preprocessing. It prevents:
  - Scaling a label-encoded column (0/1 -> 0.0 via MinMax = destroys meaning)
  - Double-encoding the same column
  - Scaling a binary column (0/1 flag)
  - Applying mean/median imputation to known categorical columns
  - Missing-value fill on columns that are already clean

Column roles:
  - 'original_numeric'   : was numeric in the raw dataset
  - 'original_categorical': was categorical (object/string) in raw dataset
  - 'label_encoded'      : was categorical, now integer via LabelEncoder
  - 'ordinal_encoded'    : was categorical, now integer via OrdinalEncoder
  - 'target_encoded'     : was categorical, now float via TargetEncoder
  - 'one_hot'            : a dummies column created by OHE (always 0/1)
  - 'binary_encoded'     : a 2-class column encoded to 0/1
  - 'scaled'             : column that has been scaled (do not re-scale)
  - 'dropped'            : column removed from the dataframe
"""

from typing import Dict, Optional, List, Set


SCALABLE_ROLES: Set[str] = {
    'original_numeric',
    'target_encoded',   # target encoding produces floats – OK to scale
}

ENCODABLE_ROLES: Set[str] = {
    'original_categorical',
}

# Roles that should NEVER be passed to a scaler
NON_SCALABLE_ROLES: Set[str] = {
    'label_encoded',
    'ordinal_encoded',
    'one_hot',
    'binary_encoded',
    'scaled',           # already scaled
    'dropped',
}


class ColumnRegistry:
    """
    Per-session column metadata store.

    Stored as:
        _registry[session_id][column_name] = {
            'role': str,
            'original_dtype': str,
            'unique_count': int,   # cached at first registration
            'nullable': bool,
        }
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Dict]] = {}

    # ── Initialisation ─────────────────────────────────────────────────────

    def init_session(self, session_id: str, df_columns_dtype: Dict[str, str],
                     df_nunique: Dict[str, int], df_nullable: Dict[str, bool]):
        """
        Register all columns from the raw uploaded dataframe.
        Call this exactly once when the dataset is first loaded.
        """
        self._registry[session_id] = {}
        for col, dtype in df_columns_dtype.items():
            if 'int' in dtype or 'float' in dtype:
                role = 'original_numeric'
            else:
                role = 'original_categorical'
            self._registry[session_id][col] = {
                'role': role,
                'original_dtype': dtype,
                'unique_count': df_nunique.get(col, -1),
                'nullable': df_nullable.get(col, False),
            }

    def copy_session(self, src_sid: str, dst_sid: str):
        """Clone registry for undo/redo snapshots."""
        if src_sid in self._registry:
            import copy
            self._registry[dst_sid] = copy.deepcopy(self._registry[src_sid])

    def clear_session(self, session_id: str):
        self._registry.pop(session_id, None)

    # ── Role queries ────────────────────────────────────────────────────────

    def get_role(self, session_id: str, column: str) -> Optional[str]:
        return self._registry.get(session_id, {}).get(column, {}).get('role')

    def get_all(self, session_id: str) -> Dict[str, Dict]:
        return self._registry.get(session_id, {})

    def is_scalable(self, session_id: str, column: str) -> bool:
        """Return True only if this column should be allowed through a scaler."""
        role = self.get_role(session_id, column)
        return role in SCALABLE_ROLES

    def is_encodable(self, session_id: str, column: str) -> bool:
        """Return True only if this column still needs encoding."""
        role = self.get_role(session_id, column)
        return role in ENCODABLE_ROLES

    def is_already_encoded(self, session_id: str, column: str) -> bool:
        role = self.get_role(session_id, column)
        return role in {'label_encoded', 'ordinal_encoded', 'target_encoded',
                        'binary_encoded', 'one_hot'}

    # ── Role mutations ──────────────────────────────────────────────────────

    def mark_label_encoded(self, session_id: str, column: str):
        self._set_role(session_id, column, 'label_encoded')

    def mark_ordinal_encoded(self, session_id: str, column: str):
        self._set_role(session_id, column, 'ordinal_encoded')

    def mark_target_encoded(self, session_id: str, column: str):
        self._set_role(session_id, column, 'target_encoded')

    def mark_binary_encoded(self, session_id: str, column: str):
        self._set_role(session_id, column, 'binary_encoded')

    def mark_one_hot(self, session_id: str, column: str):
        """Register a newly created OHE dummy column."""
        self._registry.setdefault(session_id, {})[column] = {
            'role': 'one_hot',
            'original_dtype': 'int64',
            'unique_count': 2,
            'nullable': False,
        }

    def mark_original_dropped(self, session_id: str, column: str):
        self._set_role(session_id, column, 'dropped')

    def mark_scaled(self, session_id: str, column: str):
        self._set_role(session_id, column, 'scaled')

    def mark_dropped(self, session_id: str, column: str):
        self._set_role(session_id, column, 'dropped')

    # ── Bulk helpers ────────────────────────────────────────────────────────

    def get_scalable_columns(self, session_id: str, candidates: List[str]) -> List[str]:
        """Filter *candidates* to only those that are safe to scale."""
        return [c for c in candidates if self.is_scalable(session_id, c)]

    def get_encodable_columns(self, session_id: str, candidates: List[str]) -> List[str]:
        """Filter *candidates* to only those that still need encoding."""
        return [c for c in candidates if self.is_encodable(session_id, c)]

    def get_categorical_columns(self, session_id: str) -> List[str]:
        """Return all columns currently classified as original_categorical."""
        reg = self._registry.get(session_id, {})
        return [c for c, meta in reg.items()
                if meta['role'] == 'original_categorical']

    def get_numeric_columns(self, session_id: str) -> List[str]:
        """Return all columns currently classified as original_numeric."""
        reg = self._registry.get(session_id, {})
        return [c for c, meta in reg.items()
                if meta['role'] == 'original_numeric']

    def summary(self, session_id: str) -> Dict:
        """Return a serialisable summary for the API."""
        reg = self._registry.get(session_id, {})
        by_role: Dict[str, List[str]] = {}
        for col, meta in reg.items():
            role = meta['role']
            by_role.setdefault(role, []).append(col)
        return {
            'total_columns': len(reg),
            'by_role': by_role,
            'columns': {col: meta['role'] for col, meta in reg.items()},
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _set_role(self, session_id: str, column: str, role: str):
        if session_id not in self._registry:
            self._registry[session_id] = {}
        if column not in self._registry[session_id]:
            self._registry[session_id][column] = {}
        self._registry[session_id][column]['role'] = role