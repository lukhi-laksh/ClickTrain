"""
PreprocessingEngine: Fast orchestrator using the in-memory DatasetManager.
No more disk I/O, no AuditLogger (log lives in DatasetManager).
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
            self._ready = True

    # ── Helpers ──────────────────────────────────────────
    def _cur(self, sid):
        return self.dm.get_current(sid)

    def _commit(self, sid, df, desc):
        self.dm.commit(sid, df, desc)

    # ── Init ─────────────────────────────────────────────
    def initialize_dataset(self, session_id: str, df: pd.DataFrame, dataset_name: str = None):
        self.dm.initialize_session(session_id, df, dataset_name)

    # ── Stats / history ───────────────────────────────────
    def get_dataset_stats(self, sid: str) -> Dict:
        s = self.dm.get_stats(sid)
        s['action_summary'] = {}
        return s

    def get_action_history(self, sid: str) -> List[Dict]:
        return self.dm.get_log(sid)

    def get_current_dataset(self, sid: str) -> pd.DataFrame:
        return self._cur(sid)

    # ── Missing values ───────────────────────────────────
    def analyze_missing_values(self, sid: str) -> Dict:
        df = self._cur(sid)
        result = self.nh.detect_null_values(df)
        result['total_rows'] = len(df)
        return result

    def handle_missing_values(self, sid, columns=None, strategy='mean',
                              constant_value=None, constant_string=None) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.nh.handle_missing_values(df, columns, strategy, constant_value, constant_string)
        self._commit(sid, new_df, f'Missing values: {strategy} on {len(columns) if columns else "all"} cols')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Duplicates ───────────────────────────────────────
    def analyze_duplicates(self, sid: str) -> Dict:
        return self.dh.detect_duplicates(self._cur(sid))

    def remove_duplicates(self, sid, keep='first', subset=None) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.dh.remove_duplicates(df, keep, subset)
        self._commit(sid, new_df, f'Removed duplicates (keep={keep}), {meta.get("rows_removed",0)} rows removed')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Constant columns ─────────────────────────────────
    def detect_constant_columns(self, sid: str) -> Dict:
        return self.cd.detect_constant_columns(self._cur(sid))

    def remove_constant_columns(self, sid, columns: List[str]) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.cd.remove_columns(df, columns)
        self._commit(sid, new_df, f'Removed {len(columns)} constant column(s)')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Encoding ─────────────────────────────────────────
    def label_encode(self, sid, columns: List[str]) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.em.label_encode(df, columns, sid)
        self._commit(sid, new_df, f'Label encoding on {len(meta.get("columns_encoded",[]))} cols')
        return {'shape': new_df.shape, 'metadata': meta}

    def one_hot_encode(self, sid, columns, drop_first=False, handle_binary=True) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.em.one_hot_encode(df, columns, drop_first, handle_binary, sid)
        self._commit(sid, new_df, f'One-hot encoding on {len(columns)} cols')
        return {'shape': new_df.shape, 'metadata': meta}

    def ordinal_encode(self, sid, column, categories=None, auto_order=True) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.em.ordinal_encode(df, column, categories, auto_order, sid)
        self._commit(sid, new_df, f'Ordinal encoding on {column}')
        return {'shape': new_df.shape, 'metadata': meta}

    def target_encode(self, sid, columns, target_column) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.em.target_encode(df, columns, target_column, sid)
        self._commit(sid, new_df, f'Target encoding on {len(columns)} cols')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Scaling ──────────────────────────────────────────
    def scale_features(self, sid, columns, method='standard') -> Dict:
        df = self._cur(sid)
        new_df, meta = self.sm.scale_features(df, columns, method, sid)
        self._commit(sid, new_df, f'{method.title()} scaling on {len(columns)} cols')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Outliers ─────────────────────────────────────────
    def detect_outliers(self, sid, columns, method='iqr', threshold=3.0) -> Dict:
        return self.oh.detect_outliers(self._cur(sid), columns, method, threshold)

    def handle_outliers(self, sid, columns, method='iqr', action='remove', threshold=3.0) -> Dict:
        df = self._cur(sid)
        new_df, meta = self.oh.handle_outliers(df, columns, method, action, threshold)
        self._commit(sid, new_df, f'Outliers: {action} using {method} on {len(columns)} cols')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Sampling ─────────────────────────────────────────
    def analyze_class_distribution(self, sid, target_column) -> Dict:
        return self.sh.analyze_class_distribution(self._cur(sid), target_column)

    def apply_sampling(self, sid, target_column, method='smote') -> Dict:
        df = self._cur(sid)
        if method == 'smote':
            new_df, meta = self.sh.apply_smote(df, target_column)
        elif method == 'over':
            new_df, meta = self.sh.apply_random_oversampling(df, target_column)
        else:
            new_df, meta = self.sh.apply_random_undersampling(df, target_column)
        self._commit(sid, new_df, f'{method.upper()} sampling on {target_column}')
        return {'shape': new_df.shape, 'metadata': meta}

    # ── Undo / Redo / Reset ──────────────────────────────
    def undo(self, sid) -> Dict:
        self.dm.undo(sid)
        return self.dm.get_stats(sid)

    def redo(self, sid) -> Dict:
        self.dm.redo(sid)
        return self.dm.get_stats(sid)

    def reset_to_original(self, sid) -> Dict:
        self.dm.reset(sid)
        self.em.clear_session(sid)
        self.sm.clear_session(sid)
        return self.dm.get_stats(sid)

    # ── Misc ─────────────────────────────────────────────
    def get_encoders(self, sid): return self.em.get_encoders(sid)
    def get_scalers(self, sid):  return self.sm.get_scalers(sid)

    def get_preprocessing_summary(self, sid) -> Dict:
        return {
            'dataset_stats':         self.dm.get_stats(sid),
            'action_history':        self.dm.get_log(sid),
            'encoders':              self.em.serialize_encoders(sid),
            'scalers':               self.sm.serialize_scalers(sid),
            'preprocessing_complete': bool(self.dm.get_log(sid)),
        }

    def clear_session(self, sid):
        # Sessions are in-memory only; just clear them
        for store in [self.dm._original, self.dm._current, self.dm._undo, self.dm._redo, self.dm._log, self.dm._meta]:
            store.pop(sid, None)
        self.em.clear_session(sid)
        self.sm.clear_session(sid)
