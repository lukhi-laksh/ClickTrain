"""
PreprocessingEngine: Grandmaster orchestrator.

Wires ColumnRegistry through every operation so the system KNOWS what
each column is at all times and prevents illegal operation combinations:
  - No scaling encoded columns
  - No encoding already-encoded columns
  - Registry-aware imputation (mode for encoded int columns)
  - Undo/Redo snapshots also save/restore the registry state
"""
import pandas as pd
from typing import Dict, Optional, List
from .dataset_manager import DatasetManager
from .null_value_handler import NullValueHandler
from .duplicate_handler import DuplicateHandler
from .constant_column_detector import ConstantColumnDetector
from .encoder_manager import EncoderManager
from .scaler_manager import ScalerManager
from .outlier_handler import OutlierHandler
from .sampling_handler import SamplingHandler
from .column_registry import ColumnRegistry


class PreprocessingEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_ready'):
            self.dm  = DatasetManager()
            self.nh  = NullValueHandler()
            self.dh  = DuplicateHandler()
            self.cd  = ConstantColumnDetector()
            self.em  = EncoderManager()
            self.sm  = ScalerManager()
            self.oh  = OutlierHandler()
            self.sh  = SamplingHandler()
            self.cr  = ColumnRegistry()   # ← global column registry
            # Wire registry into DatasetManager so commit/undo/redo can
            # snapshot and restore encoding roles atomically with the DataFrame
            self.dm._cr = self.cr
            self._ready = True

    # ── Helpers ──────────────────────────────────────────────────────────
    def _cur(self, sid):
        return self.dm.get_current(sid)

    def _commit(self, sid, df, desc, pre_op_registry=None):
        """Pass pre_op_registry when the caller already modified the registry."""
        self.dm.commit(sid, df, desc, pre_op_registry=pre_op_registry)

    # ── Init ─────────────────────────────────────────────────────────────
    def initialize_dataset(self, session_id: str, df: pd.DataFrame,
                           dataset_name: str = None):
        self.dm.initialize_session(session_id, df, dataset_name)
        # Register column roles in ColumnRegistry
        dtypes    = {col: str(df[col].dtype) for col in df.columns}
        nunique   = {col: int(df[col].nunique(dropna=True)) for col in df.columns}
        nullable  = {col: bool(df[col].isna().any()) for col in df.columns}
        self.cr.init_session(session_id, dtypes, nunique, nullable)

    # ── Stats / history ───────────────────────────────────────────────────
    def get_dataset_stats(self, sid: str) -> Dict:
        s = self.dm.get_stats(sid)
        s['action_summary'] = {}
        s['column_registry'] = self.cr.summary(sid)
        return s

    def get_action_history(self, sid: str) -> List[Dict]:
        return self.dm.get_log(sid)

    def quick_summary(self, sid: str) -> Dict:
        """
        Single fast pass over the current DataFrame to compute everything
        needed for the three topbar badges (nulls, duplicates, constant cols).
        One DataFrame read, all vectorised — no repeated disk I/O.
        """
        df = self._cur(sid)

        # ── Missing values ────────────────────────────────────────────────
        total_nulls = int(df.isnull().sum().sum())

        # ── Duplicates ────────────────────────────────────────────────────
        # rows_to_remove = rows actually deleted by drop_duplicates(keep='first')
        rows_to_remove = int(df.duplicated(keep='first').sum())

        # ── Constant columns ──────────────────────────────────────────────
        # A column is constant when it has <= 1 unique non-null value
        unique_counts = df.nunique()
        constant_col_count = int((unique_counts <= 1).sum())

        return {
            'total_nulls':      total_nulls,
            'rows_to_remove':   rows_to_remove,
            'constant_columns': constant_col_count,
        }

    def get_current_dataset(self, sid: str) -> pd.DataFrame:
        return self._cur(sid)

    # ── Missing values ───────────────────────────────────────────────────
    def analyze_missing_values(self, sid: str) -> Dict:
        df = self._cur(sid)
        result = self.nh.detect_null_values(df)
        result['total_rows'] = len(df)
        return result

    def handle_missing_values(
        self, sid: str,
        columns=None,
        strategy: str = 'mean',
        constant_value=None,
        constant_string=None,
        min_valid_ratio: float = 0.0,
    ) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.nh.handle_missing_values(
            df, columns, strategy, constant_value, constant_string,
            min_valid_ratio=min_valid_ratio,
            registry=self.cr, session_id=sid,
        )
        col_count = len(meta.get('columns_processed', columns or []))
        self._commit(sid, new_df,
                     f'Missing values: {strategy} on {col_count} col(s)')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Duplicates ───────────────────────────────────────────────────────
    def analyze_duplicates(self, sid: str) -> Dict:
        return self.dh.detect_duplicates(self._cur(sid))

    def remove_duplicates(self, sid: str, keep='first', subset=None) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.dh.remove_duplicates(df, keep, subset)
        self._commit(sid, new_df,
                     f'Removed duplicates (keep={keep}), '
                     f'{meta.get("rows_removed", 0)} rows removed')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Constant columns ─────────────────────────────────────────────────
    def detect_constant_columns(self, sid: str) -> Dict:
        return self.cd.detect_constant_columns(self._cur(sid))

    def remove_constant_columns(self, sid: str, columns: List[str]) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.cd.remove_columns(df, columns)
        for col in columns:
            self.cr.mark_dropped(sid, col)
        self._commit(sid, new_df,
                     f'Removed {len(columns)} constant column(s)')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Encoding ─────────────────────────────────────────────────────────

    def label_encode(self, sid: str, columns: List[str]) -> Dict:
        df = self._cur(sid)
        pre_reg = self.dm._snap_registry(sid)   # snapshot BEFORE registry is mutated
        new_df, meta = self.em.label_encode(df, columns, sid, registry=self.cr)
        encoded = meta.get('columns_encoded', [])
        self._commit(sid, new_df,
                     f'Label encoding on {len(encoded)} col(s)',
                     pre_op_registry=pre_reg)
        return {'shape': new_df.shape, 'metadata': meta}

    def one_hot_encode(self, sid: str, columns: List[str],
                       drop_first: bool = False,
                       handle_binary: bool = True,
                       max_categories: int = 50) -> Dict:
        df = self._cur(sid)
        pre_reg = self.dm._snap_registry(sid)   # snapshot BEFORE registry is mutated
        new_df, meta = self.em.one_hot_encode(
            df, columns, drop_first, handle_binary, sid,
            registry=self.cr, max_categories=max_categories
        )
        self._commit(sid, new_df,
                     f'One-hot encoding on {len(columns)} col(s)',
                     pre_op_registry=pre_reg)
        return {'shape': new_df.shape, 'metadata': meta}

    def ordinal_encode(self, sid: str, column: str,
                       categories=None, auto_order: bool = True) -> Dict:
        df = self._cur(sid)
        pre_reg = self.dm._snap_registry(sid)   # snapshot BEFORE registry is mutated
        new_df, meta = self.em.ordinal_encode(
            df, column, categories, auto_order, sid, registry=self.cr
        )
        self._commit(sid, new_df,
                     f'Ordinal encoding on {column}',
                     pre_op_registry=pre_reg)
        return {'shape': new_df.shape, 'metadata': meta}

    def target_encode(self, sid: str, columns: List[str],
                      target_column: str,
                      smoothing: float = 10.0) -> Dict:
        df = self._cur(sid)
        pre_reg = self.dm._snap_registry(sid)   # snapshot BEFORE registry is mutated
        new_df, meta = self.em.target_encode(
            df, columns, target_column, sid,
            registry=self.cr, smoothing=smoothing
        )
        self._commit(sid, new_df,
                     f'Target encoding on {len(columns)} col(s)',
                     pre_op_registry=pre_reg)
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Scaling ──────────────────────────────────────────────────────────

    def scale_features(self, sid: str, columns: List[str],
                       method: str = 'standard') -> Dict:
        df = self._cur(sid)
        new_df, meta = self.sm.scale_features(
            df, columns, method, sid, registry=self.cr
        )
        scaled = meta.get('columns_scaled', [])
        self._commit(sid, new_df,
                     f'{method.title()} scaling on {len(scaled)} col(s)')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Column registry status ────────────────────────────────────────────

    def get_column_registry(self, sid: str) -> Dict:
        """Return the full column role summary for the session."""
        return self.cr.summary(sid)

    def get_scalable_columns(self, sid: str) -> List[str]:
        """Return only the columns that are currently safe to scale."""
        df = self._cur(sid)
        return self.cr.get_scalable_columns(sid, list(df.columns))

    def get_encodable_columns(self, sid: str) -> List[str]:
        """Return only the columns that still need encoding (original categorical)."""
        df = self._cur(sid)
        return self.cr.get_encodable_columns(sid, list(df.columns))

    # ── Outliers ─────────────────────────────────────────────────────────

    def detect_outliers(self, sid: str, columns: List[str],
                        method: str = 'iqr', threshold: float = 3.0) -> Dict:
        return self.oh.detect_outliers(self._cur(sid), columns, method, threshold)

    def handle_outliers(self, sid: str, columns: List[str],
                        method: str = 'iqr', action: str = 'remove',
                        threshold: float = 3.0) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.oh.handle_outliers(df, columns, method, action, threshold)
        self._commit(sid, new_df,
                     f'Outliers: {action} using {method} on {len(columns)} col(s)')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Sampling ─────────────────────────────────────────────────────────

    def analyze_class_distribution(self, sid: str, target_column: str) -> Dict:
        return self.sh.analyze_class_distribution(self._cur(sid), target_column)

    def apply_sampling(self, sid: str, target_column: str,
                       method: str = 'smote') -> Dict:
        df = self._cur(sid)
        if method == 'smote':
            new_df, meta = self.sh.apply_smote(df, target_column)
        elif method == 'over':
            new_df, meta = self.sh.apply_random_oversampling(df, target_column)
        else:
            new_df, meta = self.sh.apply_random_undersampling(df, target_column)
        self._commit(sid, new_df,
                     f'{method.upper()} sampling on {target_column}')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Undo / Redo / Reset ──────────────────────────────────────────────

    def undo(self, sid: str) -> Dict:
        self.dm.undo(sid)
        return self.dm.get_stats(sid)

    def redo(self, sid: str) -> Dict:
        self.dm.redo(sid)
        return self.dm.get_stats(sid)

    def reset_to_original(self, sid: str) -> Dict:
        self.dm.reset(sid)
        self.em.clear_session(sid)
        self.sm.clear_session(sid)
        # Re-init registry from original dataset
        df = self._cur(sid)
        dtypes   = {col: str(df[col].dtype) for col in df.columns}
        nunique  = {col: int(df[col].nunique(dropna=True)) for col in df.columns}
        nullable = {col: bool(df[col].isna().any()) for col in df.columns}
        self.cr.init_session(sid, dtypes, nunique, nullable)
        return self.dm.get_stats(sid)

    # ── Misc ─────────────────────────────────────────────────────────────

    def get_encoders(self, sid: str): return self.em.get_encoders(sid)
    def get_scalers(self, sid: str):  return self.sm.get_scalers(sid)

    def get_preprocessing_summary(self, sid: str) -> Dict:
        return {
            'dataset_stats':          self.dm.get_stats(sid),
            'action_history':         self.dm.get_log(sid),
            'encoders':               self.em.serialize_encoders(sid),
            'scalers':                self.sm.serialize_scalers(sid),
            'column_registry':        self.cr.summary(sid),
            'scalable_columns':       self.get_scalable_columns(sid),
            'encodable_columns':      self.get_encodable_columns(sid),
            'preprocessing_complete': bool(self.dm.get_log(sid)),
        }

    def clear_session(self, sid: str):
        for store in [
            self.dm._original, self.dm._current,
            self.dm._undo, self.dm._redo,
            self.dm._log, self.dm._meta
        ]:
            store.pop(sid, None)
        self.em.clear_session(sid)
        self.sm.clear_session(sid)
        self.cr.clear_session(sid)
