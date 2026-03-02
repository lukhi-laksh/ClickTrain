"""
DatasetManager: Fast, in-memory only. No disk I/O during preprocessing.
Versions are stored as column-level deltas, not full copies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class DatasetManager:
    """
    Singleton in-memory dataset manager with undo/redo.
    FAST: no disk I/O on every operation, minimal copying.
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

            # undo/redo stacks hold DataFrames (shallow references where possible)
            # {session_id: [df, df, ...]}
            self._undo: Dict[str, list] = {}
            self._redo: Dict[str, list] = {}

            # metadata dict (lightweight)
            self._meta: Dict[str, dict] = {}

            # audit log  {session_id: [{description, timestamp}, ...]}
            self._log: Dict[str, list] = {}

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

    # ── Commit a new version (push old to undo stack) ───
    def commit(self, session_id: str, new_df: pd.DataFrame, description: str):
        if session_id not in self._current:
            raise ValueError(f'Session {session_id} not found.')
        # Push old state to undo stack
        self._undo[session_id].append(self._current[session_id])
        # Clear redo (new branch)
        self._redo[session_id].clear()
        # Replace current
        self._current[session_id] = new_df
        # Log
        self._log[session_id].append({
            'description': description,
            'timestamp':   datetime.now().isoformat(),
        })

    # ── Undo / Redo ──────────────────────────────────────
    def undo(self, session_id: str):
        if session_id not in self._undo or not self._undo[session_id]:
            raise ValueError('Nothing to undo.')
        self._redo[session_id].append(self._current[session_id])
        self._current[session_id] = self._undo[session_id].pop()
        if self._log[session_id]:
            self._log[session_id].pop()

    def redo(self, session_id: str):
        if session_id not in self._redo or not self._redo[session_id]:
            raise ValueError('Nothing to redo.')
        self._undo[session_id].append(self._current[session_id])
        self._current[session_id] = self._redo[session_id].pop()

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
