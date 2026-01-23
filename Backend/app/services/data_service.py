import pandas as pd
from typing import Dict

class DataService:
    """
    Service to manage dataset storage in memory for multiple users.
    Uses session IDs to isolate user data.
    Singleton pattern to ensure all routes use the same data store.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not DataService._initialized:
            self.data_store: Dict[str, pd.DataFrame] = {}
            self.processed_data_store: Dict[str, Dict] = {}  # For preprocessed data splits
            DataService._initialized = True

    def store_data(self, session_id: str, df: pd.DataFrame):
        """Store the uploaded dataset"""
        self.data_store[session_id] = df.copy()

    def get_data(self, session_id: str) -> pd.DataFrame:
        """Retrieve the dataset for a session"""
        if session_id not in self.data_store:
            raise ValueError("Session not found. Please upload a dataset first.")
        return self.data_store[session_id]

    def store_processed_data(self, session_id: str, processed_data: Dict):
        """Store preprocessed data splits"""
        self.processed_data_store[session_id] = processed_data

    def get_processed_data(self, session_id: str) -> Dict:
        """Retrieve preprocessed data splits"""
        if session_id not in self.processed_data_store:
            raise ValueError("No preprocessed data found. Please run preprocessing first.")
        return self.processed_data_store[session_id]

    def clear_session(self, session_id: str):
        """Clear data for a session"""
        if session_id in self.data_store:
            del self.data_store[session_id]
        if session_id in self.processed_data_store:
            del self.processed_data_store[session_id]