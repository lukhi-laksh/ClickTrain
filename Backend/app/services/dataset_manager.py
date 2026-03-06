"""
DatasetManager: Fast, in-memory only. No disk I/O during preprocessing.
Undo/redo stacks store (DataFrame, registry_snapshot, log_entry) tuples
so that the ColumnRegistry is fully reverted alongside the data.
"""
import copy
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class DatasetManager:
    """
    Singleton in-memory dataset manager with undo/redo.
    FAST: no disk I/O on every operation, minimal copying.

    Each undo/redo stack entry is a 3-tuple:
        (DataFrame, registry_dict_snapshot, log_entry_dict)
    This ensures that encoding roles (ColumnRegistry) and history are
    ALL reverted/re-applied together when the user presses Undo or Redo.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DatasetManager._initialized:
            # {session_id: DataFrame}  — single live copy each
            self._original: Dict[str, pd.DataFrame] = {}
            self._current:  Dict[str, pd.DataFrame] = {}

            # undo/redo stacks hold (DataFrame, registry_snapshot, log_entry)
            # {session_id: [(df, reg_snap, entry), ...]}
            self._undo: Dict[str, list] = {}
            self._redo: Dict[str, list] = {}

            # metadata dict (lightweight)
            self._meta: Dict[str, dict] = {}

            # audit log  {session_id: [{description, timestamp}, ...]}
            self._log: Dict[str, list] = {}

            # Reference to ColumnRegistry — wired in by PreprocessingEngine
            self._cr = None

            DatasetManager._initialized = True

    # ── Session init ────────────────────────────────────
    def initialize_session(self, session_id: str, df: pd.DataFrame, dataset_name: str = None):
        self._original[session_id] = df  # keep reference, don't copy
        self._current[session_id]  = df.copy()
        self._undo[session_id]     = []
        self._redo[session_id]     = []
        self._log[session_id]      = []
        self._meta[session_id] = {
            'dataset_name':    dataset_name or f'dataset_{session_id[:8]}',
            'original_rows':   df.shape[0],
            'original_cols':   df.shape[1],
            'original_cols_list': list(df.columns),
            'created_at':      datetime.now().isoformat(),
        }

    # ── Read current (NO copy — callers must not mutate) ─
    def get_current(self, session_id: str) -> pd.DataFrame:
        if session_id not in self._current:
            raise ValueError(f'Session {session_id} not found. Please upload a dataset first.')
        return self._current[session_id]

    def get_original(self, session_id: str) -> pd.DataFrame:
        if session_id not in self._original:
            raise ValueError(f'Session {session_id} not found.')
        return self._original[session_id]

    # ── Registry snapshot helpers ────────────────────────
    def _snap_registry(self, session_id: str) -> Optional[dict]:
        """Take a deep copy of the current registry state for this session."""
        if self._cr is None:
            return None
        reg = self._cr._registry.get(session_id)
        return copy.deepcopy(reg) if reg is not None else None

    def _restore_registry(self, session_id: str, snapshot: Optional[dict]):
        """Restore a previously snapshotted registry state."""
        if self._cr is None or snapshot is None:
            return
        self._cr._registry[session_id] = snapshot

    # ── Commit a new version (push old to undo stack) ───
    def commit(self, session_id: str, new_df: pd.DataFrame, description: str,
               pre_op_registry=None):
        """
        Commit a new dataframe version.

        pre_op_registry: registry snapshot taken BEFORE the operation ran.
          Pass this when your operation modifies the ColumnRegistry before
          calling commit (e.g. all encoding operations).  When None, the
          snapshot is taken inside commit — which is AFTER the operation
          already changed the registry, making undo restore the wrong roles.
        """
        if session_id not in self._current:
            raise ValueError(f'Session {session_id} not found.')

        # Use the caller-supplied pre-operation snapshot when available;
        # fall back to snapshotting now (for operations that don't touch
        # the registry, e.g. drop-duplicates, fill-nulls, scale, etc.).
        reg_snap = pre_op_registry if pre_op_registry is not None \
            else self._snap_registry(session_id)

        log_entry = {
            'description': description,
            'timestamp':   datetime.now().isoformat(),
        }

        # Push (old_df, pre-op_registry, new_log_entry) onto undo stack
        self._undo[session_id].append(
            (self._current[session_id], reg_snap, log_entry)
        )
        # Clear redo (new action invalidates redo history)
        self._redo[session_id].clear()
        # Replace current dataframe
        self._current[session_id] = new_df
        # Append to visible log
        self._log[session_id].append(log_entry)

    # ── Undo / Redo ──────────────────────────────────────
    def undo(self, session_id: str):
        if session_id not in self._undo or not self._undo[session_id]:
            raise ValueError('Nothing to undo.')

        # Capture current state to push onto redo stack
        cur_reg_snap  = self._snap_registry(session_id)
        cur_log_entry = self._log[session_id][-1] if self._log[session_id] else None

        self._redo[session_id].append(
            (self._current[session_id], cur_reg_snap, cur_log_entry)
        )

        # Pop previous state
        prev_df, prev_reg, _ = self._undo[session_id].pop()

        # Restore DataFrame + registry to previous state
        self._current[session_id] = prev_df
        self._restore_registry(session_id, prev_reg)

        # Remove last log entry
        if self._log[session_id]:
            self._log[session_id].pop()

    def redo(self, session_id: str):
        if session_id not in self._redo or not self._redo[session_id]:
            raise ValueError('Nothing to redo.')

        # Capture current state to push onto undo stack
        cur_reg_snap  = self._snap_registry(session_id)
        cur_log_entry = self._log[session_id][-1] if self._log[session_id] else None

        self._undo[session_id].append(
            (self._current[session_id], cur_reg_snap, cur_log_entry)
        )

        # Pop the redo state
        next_df, next_reg, next_log = self._redo[session_id].pop()

        # Restore DataFrame + registry
        self._current[session_id] = next_df
        self._restore_registry(session_id, next_reg)

        # Restore the log entry that was undone
        if next_log:
            self._log[session_id].append(next_log)

    # ── Reset ────────────────────────────────────────────
    def reset(self, session_id: str):
        self._current[session_id] = self._original[session_id].copy()
        self._undo[session_id].clear()
        self._redo[session_id].clear()
        self._log[session_id].clear()

    # ── Stats ────────────────────────────────────────────
    def get_stats(self, session_id: str) -> dict:
        if session_id not in self._current:
            raise ValueError(f'Session {session_id} not found.')
        cur = self._current[session_id]
        m   = self._meta[session_id]
        return {
            'dataset_name':     m['dataset_name'],
            'current_rows':     cur.shape[0],
            'current_columns':  cur.shape[1],
            'original_rows':    m['original_rows'],
            'original_columns': m['original_cols'],
            'can_undo':         bool(self._undo.get(session_id)),
            'can_redo':         bool(self._redo.get(session_id)),
        }

    def get_log(self, session_id: str) -> list:
        return self._log.get(session_id, [])

    def get_current_dataset(self, session_id: str) -> pd.DataFrame:
        """Alias used by export endpoint."""
        return self.get_current(session_id)
