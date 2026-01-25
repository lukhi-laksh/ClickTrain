"""
DatasetManager: Manages dataset versions, original dataset preservation, and metadata tracking.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import copy


class DatasetVersion:
    """Represents a version of the dataset with metadata."""
    
    def __init__(self, df: pd.DataFrame, version_id: int, action: str, metadata: Dict):
        self.df = df.copy()
        self.version_id = version_id
        self.action = action
        self.metadata = metadata
        self.timestamp = datetime.now().isoformat()
        self.shape = df.shape
        self.columns = list(df.columns)
        self.dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}


class DatasetManager:
    """
    Manages dataset versions with undo/redo support.
    Maintains original dataset (immutable) and current dataset (mutable).
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not DatasetManager._initialized:
            # Original datasets (immutable)
            self.original_datasets: Dict[str, pd.DataFrame] = {}
            
            # Current datasets (mutable)
            self.current_datasets: Dict[str, pd.DataFrame] = {}
            
            # Version history: {session_id: [DatasetVersion]}
            self.version_history: Dict[str, list] = {}
            
            # Undo/Redo stacks: {session_id: {'undo': [...], 'redo': [...]}}
            self.version_stacks: Dict[str, Dict[str, list]] = {}
            
            # Metadata: {session_id: {'original_shape': ..., 'original_columns': ...}}
            self.metadata: Dict[str, Dict] = {}
            
            DatasetManager._initialized = True
    
    def initialize_session(self, session_id: str, df: pd.DataFrame, dataset_name: str = None):
        """
        Initialize a new session with an original dataset.
        This is called when a dataset is first uploaded.
        """
        # Store original (immutable)
        self.original_datasets[session_id] = df.copy()
        
        # Initialize current dataset
        self.current_datasets[session_id] = df.copy()
        
        # Initialize version history
        self.version_history[session_id] = []
        
        # Initialize stacks
        self.version_stacks[session_id] = {'undo': [], 'redo': []}
        
        # Store metadata
        self.metadata[session_id] = {
            'dataset_name': dataset_name or f"dataset_{session_id}",
            'original_shape': df.shape,
            'original_columns': list(df.columns),
            'original_dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'created_at': datetime.now().isoformat()
        }
        
        # Create initial version
        initial_version = DatasetVersion(
            df=df.copy(),
            version_id=0,
            action="initial_upload",
            metadata={'description': 'Original dataset'}
        )
        self.version_history[session_id].append(initial_version)
    
    def get_original(self, session_id: str) -> pd.DataFrame:
        """Get the original (immutable) dataset."""
        if session_id not in self.original_datasets:
            raise ValueError(f"Session {session_id} not found. Please upload a dataset first.")
        return self.original_datasets[session_id].copy()
    
    def get_current(self, session_id: str) -> pd.DataFrame:
        """Get the current (mutable) dataset."""
        if session_id not in self.current_datasets:
            raise ValueError(f"Session {session_id} not found. Please upload a dataset first.")
        return self.current_datasets[session_id].copy()
    
    def create_version(self, session_id: str, df: pd.DataFrame, action: str, metadata: Dict) -> int:
        """
        Create a new version of the dataset.
        Returns the version ID.
        """
        if session_id not in self.current_datasets:
            raise ValueError(f"Session {session_id} not found.")
        
        # Update current dataset
        self.current_datasets[session_id] = df.copy()
        
        # Get next version ID
        version_id = len(self.version_history[session_id])
        
        # Create version
        version = DatasetVersion(
            df=df.copy(),
            version_id=version_id,
            action=action,
            metadata=metadata
        )
        
        # Add to history
        self.version_history[session_id].append(version)
        
        # Push to undo stack
        self.version_stacks[session_id]['undo'].append(version_id)
        
        # Clear redo stack (new action invalidates redo)
        self.version_stacks[session_id]['redo'] = []
        
        return version_id
    
    def get_version(self, session_id: str, version_id: int) -> DatasetVersion:
        """Get a specific version by ID."""
        if session_id not in self.version_history:
            raise ValueError(f"Session {session_id} not found.")
        
        for version in self.version_history[session_id]:
            if version.version_id == version_id:
                return version
        
        raise ValueError(f"Version {version_id} not found for session {session_id}.")
    
    def undo(self, session_id: str) -> Optional[DatasetVersion]:
        """
        Undo the last action.
        Returns the restored version or None if nothing to undo.
        """
        if session_id not in self.version_stacks:
            raise ValueError(f"Session {session_id} not found.")
        
        undo_stack = self.version_stacks[session_id]['undo']
        redo_stack = self.version_stacks[session_id]['redo']
        
        if len(undo_stack) <= 1:  # Can't undo initial version
            return None
        
        # Pop from undo stack
        current_version_id = undo_stack.pop()
        redo_stack.append(current_version_id)
        
        # Get previous version
        previous_version_id = undo_stack[-1]
        previous_version = self.get_version(session_id, previous_version_id)
        
        # Restore dataset
        self.current_datasets[session_id] = previous_version.df.copy()
        
        return previous_version
    
    def redo(self, session_id: str) -> Optional[DatasetVersion]:
        """
        Redo the last undone action.
        Returns the restored version or None if nothing to redo.
        """
        if session_id not in self.version_stacks:
            raise ValueError(f"Session {session_id} not found.")
        
        undo_stack = self.version_stacks[session_id]['undo']
        redo_stack = self.version_stacks[session_id]['redo']
        
        if len(redo_stack) == 0:
            return None
        
        # Pop from redo stack
        version_id = redo_stack.pop()
        undo_stack.append(version_id)
        
        # Get version
        version = self.get_version(session_id, version_id)
        
        # Restore dataset
        self.current_datasets[session_id] = version.df.copy()
        
        return version
    
    def reset_to_original(self, session_id: str) -> DatasetVersion:
        """Reset dataset to original state."""
        original_df = self.get_original(session_id)
        
        # Reset current
        self.current_datasets[session_id] = original_df.copy()
        
        # Reset stacks
        self.version_stacks[session_id] = {
            'undo': [0],  # Only initial version
            'redo': []
        }
        
        # Get initial version
        return self.get_version(session_id, 0)
    
    def get_stats(self, session_id: str) -> Dict:
        """Get current dataset statistics."""
        if session_id not in self.current_datasets:
            raise ValueError(f"Session {session_id} not found.")
        
        df = self.current_datasets[session_id]
        original_df = self.original_datasets[session_id]
        
        return {
            'current_shape': df.shape,
            'current_rows': df.shape[0],
            'current_columns': df.shape[1],
            'original_shape': original_df.shape,
            'original_rows': original_df.shape[0],
            'original_columns': original_df.shape[1],
            'version_count': len(self.version_history[session_id]),
            'can_undo': len(self.version_stacks[session_id]['undo']) > 1,
            'can_redo': len(self.version_stacks[session_id]['redo']) > 0,
            'dataset_name': self.metadata[session_id]['dataset_name']
        }
    
    def get_action_history(self, session_id: str) -> list:
        """Get list of actions applied."""
        if session_id not in self.version_history:
            return []
        
        history = []
        for version in self.version_history[session_id]:
            if version.version_id > 0:  # Skip initial version
                history.append({
                    'version_id': version.version_id,
                    'action': version.action,
                    'timestamp': version.timestamp,
                    'metadata': version.metadata,
                    'shape': version.shape
                })
        
        return history
    
    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        if session_id in self.original_datasets:
            del self.original_datasets[session_id]
        if session_id in self.current_datasets:
            del self.current_datasets[session_id]
        if session_id in self.version_history:
            del self.version_history[session_id]
        if session_id in self.version_stacks:
            del self.version_stacks[session_id]
        if session_id in self.metadata:
            del self.metadata[session_id]
