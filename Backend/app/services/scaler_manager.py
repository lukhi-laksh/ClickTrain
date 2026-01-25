"""
ScalerManager: Manages feature scaling operations.
Supports StandardScaler, MinMaxScaler, and RobustScaler.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import base64


class ScalerManager:
    """
    Manages scaling operations with persistence support.
    Stores scaler objects for reuse during training.
    """
    
    def __init__(self):
        # Store scalers per session: {session_id: {column: scaler_object}}
        self.scalers: Dict[str, Dict[str, any]] = {}
    
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'standard',
        session_id: str = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale specified numerical columns.
        
        Args:
            method: 'standard', 'minmax', or 'robust'
        
        Returns:
            Tuple of (scaled_df, metadata)
        """
        df_scaled = df.copy()
        metadata = {
            'method': method,
            'columns_scaled': [],
            'before_stats': {},
            'after_stats': {},
            'scalers': {}
        }
        
        if session_id not in self.scalers:
            self.scalers[session_id] = {}
        
        for col in columns:
            if col not in df_scaled.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                continue  # Skip non-numeric columns
            
            # Get before stats
            before_mean = df_scaled[col].mean()
            before_std = df_scaled[col].std()
            before_min = df_scaled[col].min()
            before_max = df_scaled[col].max()
            
            metadata['before_stats'][col] = {
                'mean': float(before_mean) if not pd.isna(before_mean) else None,
                'std': float(before_std) if not pd.isna(before_std) else None,
                'min': float(before_min) if not pd.isna(before_min) else None,
                'max': float(before_max) if not pd.isna(before_max) else None
            }
            
            # Create and fit scaler
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Reshape for sklearn
            col_data = df_scaled[col].values.reshape(-1, 1)
            scaled_data = scaler.fit_transform(col_data)
            
            # Update dataframe
            df_scaled[col] = scaled_data.flatten()
            
            # Store scaler
            self.scalers[session_id][col] = scaler
            
            # Get after stats
            after_mean = df_scaled[col].mean()
            after_std = df_scaled[col].std()
            after_min = df_scaled[col].min()
            after_max = df_scaled[col].max()
            
            metadata['after_stats'][col] = {
                'mean': float(after_mean) if not pd.isna(after_mean) else None,
                'std': float(after_std) if not pd.isna(after_std) else None,
                'min': float(after_min) if not pd.isna(after_min) else None,
                'max': float(after_max) if not pd.isna(after_max) else None
            }
            
            metadata['columns_scaled'].append(col)
            
            # Store scaler parameters
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                metadata['scalers'][col] = {
                    'type': method,
                    'mean': float(scaler.mean_[0]) if len(scaler.mean_) > 0 else None,
                    'scale': float(scaler.scale_[0]) if len(scaler.scale_) > 0 else None
                }
            elif hasattr(scaler, 'min_') and hasattr(scaler, 'data_min_'):
                metadata['scalers'][col] = {
                    'type': method,
                    'min': float(scaler.data_min_[0]) if len(scaler.data_min_) > 0 else None,
                    'max': float(scaler.data_max_[0]) if len(scaler.data_max_) > 0 else None,
                    'scale': float(scaler.scale_[0]) if len(scaler.scale_) > 0 else None,
                    'min_shift': float(scaler.min_[0]) if len(scaler.min_) > 0 else None
                }
        
        return df_scaled, metadata
    
    def get_scalers(self, session_id: str) -> Dict:
        """Get all scalers for a session."""
        return self.scalers.get(session_id, {})
    
    def serialize_scalers(self, session_id: str) -> Dict:
        """
        Serialize scalers for persistence.
        Returns a dictionary with serialized scaler data.
        """
        if session_id not in self.scalers:
            return {}
        
        serialized = {}
        for col, scaler in self.scalers[session_id].items():
            scaler_bytes = pickle.dumps(scaler)
            scaler_b64 = base64.b64encode(scaler_bytes).decode('utf-8')
            serialized[col] = {
                'type': type(scaler).__name__,
                'data': scaler_b64
            }
        
        return serialized
    
    def clear_session(self, session_id: str):
        """Clear scalers for a session."""
        if session_id in self.scalers:
            del self.scalers[session_id]
