"""
EncoderManager: Manages all categorical encoding operations.
Supports Label, One-Hot, Ordinal, and Target encoding.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pickle
import base64


class EncoderManager:
    """
    Manages encoding operations with persistence support.
    Stores encoder objects for reuse during training.
    """
    
    def __init__(self):
        # Store encoders per session: {session_id: {column: encoder_object}}
        self.encoders: Dict[str, Dict[str, any]] = {}
    
    def label_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        session_id: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Label Encoding to specified columns.
        
        Returns:
            Tuple of (encoded_df, metadata)
        """
        df_encoded = df.copy()
        metadata = {
            'method': 'label',
            'columns_encoded': [],
            'encoders': {}
        }
        
        if session_id not in self.encoders:
            self.encoders[session_id] = {}
        
        for col in columns:
            if col not in df_encoded.columns:
                continue
            
            if not pd.api.types.is_object_dtype(df_encoded[col]) and \
               not pd.api.types.is_categorical_dtype(df_encoded[col]):
                continue  # Skip non-categorical columns
            
            # Create and fit encoder
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            
            # Store encoder
            self.encoders[session_id][col] = encoder
            metadata['columns_encoded'].append(col)
            metadata['encoders'][col] = {
                'type': 'label',
                'classes': encoder.classes_.tolist()
            }
        
        return df_encoded, metadata
    
    def one_hot_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_first: bool = False,
        handle_binary: bool = True,
        session_id: str = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply One-Hot Encoding to specified columns.
        
        Args:
            drop_first: Drop first category to avoid multicollinearity
            handle_binary: If True, binary columns use Label Encoding instead
        
        Returns:
            Tuple of (encoded_df, metadata)
        """
        df_encoded = df.copy()
        metadata = {
            'method': 'one_hot',
            'columns_encoded': [],
            'new_columns': [],
            'drop_first': drop_first,
            'columns_dropped': []
        }
        
        for col in columns:
            if col not in df_encoded.columns:
                continue
            
            # Check if binary column
            unique_count = df_encoded[col].nunique()
            
            if handle_binary and unique_count == 2:
                # Use Label Encoding for binary columns
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                metadata['columns_encoded'].append({
                    'original': col,
                    'method': 'label',
                    'reason': 'binary_column'
                })
            else:
                # One-Hot Encoding
                dummies = pd.get_dummies(
                    df_encoded[col],
                    prefix=col,
                    drop_first=drop_first,
                    dtype=int
                )
                
                # Get new column names
                new_cols = dummies.columns.tolist()
                metadata['new_columns'].extend(new_cols)
                
                # Drop original column and add dummies
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                metadata['columns_encoded'].append({
                    'original': col,
                    'method': 'one_hot',
                    'new_columns': new_cols
                })
                
                if drop_first:
                    metadata['columns_dropped'].append(new_cols[0] if new_cols else None)
        
        metadata['total_new_columns'] = len(metadata['new_columns'])
        
        return df_encoded, metadata
    
    def ordinal_encode(
        self,
        df: pd.DataFrame,
        column: str,
        categories: Optional[List] = None,
        auto_order: bool = True,
        session_id: str = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Ordinal Encoding to a single column.
        
        Args:
            categories: Manual category order (if provided, auto_order is ignored)
            auto_order: Auto-detect order from data
        
        Returns:
            Tuple of (encoded_df, metadata)
        """
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in dataframe.")
        
        df_encoded = df.copy()
        
        # Determine categories
        if categories:
            category_order = categories
        elif auto_order:
            # Auto-detect: sort unique values
            unique_vals = df_encoded[column].unique()
            category_order = sorted([str(v) for v in unique_vals if pd.notna(v)])
        else:
            # Use existing order
            category_order = df_encoded[column].unique().tolist()
        
        # Create mapping
        category_map = {cat: idx for idx, cat in enumerate(category_order)}
        
        # Apply encoding
        df_encoded[column] = df_encoded[column].astype(str).map(category_map)
        
        # Handle any unmapped values (shouldn't happen, but safety)
        df_encoded[column] = df_encoded[column].fillna(-1)
        df_encoded[column] = df_encoded[column].astype(int)
        
        metadata = {
            'method': 'ordinal',
            'column': column,
            'category_order': category_order,
            'category_map': category_map,
            'auto_order': auto_order and categories is None
        }
        
        # Store encoder info
        if session_id:
            if session_id not in self.encoders:
                self.encoders[session_id] = {}
            self.encoders[session_id][column] = {
                'type': 'ordinal',
                'category_map': category_map
            }
        
        return df_encoded, metadata
    
    def target_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_column: str,
        session_id: str = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Target Encoding (Mean Encoding) to specified columns.
        
        WARNING: This can cause data leakage if not done carefully.
        Should be used with cross-validation.
        
        Returns:
            Tuple of (encoded_df, metadata)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found.")
        
        df_encoded = df.copy()
        metadata = {
            'method': 'target',
            'target_column': target_column,
            'columns_encoded': [],
            'encodings': {}
        }
        
        if session_id not in self.encoders:
            self.encoders[session_id] = {}
        
        for col in columns:
            if col not in df_encoded.columns or col == target_column:
                continue
            
            # Calculate mean target value per category
            target_means = df_encoded.groupby(col)[target_column].mean().to_dict()
            
            # Apply encoding
            df_encoded[col] = df_encoded[col].map(target_means)
            
            # Fill any missing values with overall mean
            overall_mean = df_encoded[target_column].mean()
            df_encoded[col] = df_encoded[col].fillna(overall_mean)
            
            # Store encoding
            self.encoders[session_id][col] = {
                'type': 'target',
                'target_means': target_means,
                'overall_mean': float(overall_mean)
            }
            
            metadata['columns_encoded'].append(col)
            metadata['encodings'][col] = {
                'unique_categories': len(target_means),
                'overall_mean': float(overall_mean)
            }
        
        metadata['warning'] = 'Target encoding can cause data leakage. Use with cross-validation.'
        
        return df_encoded, metadata
    
    def get_encoders(self, session_id: str) -> Dict:
        """Get all encoders for a session."""
        return self.encoders.get(session_id, {})
    
    def serialize_encoders(self, session_id: str) -> Dict:
        """
        Serialize encoders for persistence.
        Returns a dictionary with serialized encoder data.
        """
        if session_id not in self.encoders:
            return {}
        
        serialized = {}
        for col, encoder_data in self.encoders[session_id].items():
            if isinstance(encoder_data, dict):
                # Already a dict (target, ordinal)
                serialized[col] = encoder_data
            else:
                # Sklearn encoder object
                encoder_bytes = pickle.dumps(encoder_data)
                encoder_b64 = base64.b64encode(encoder_bytes).decode('utf-8')
                serialized[col] = {
                    'type': 'sklearn',
                    'data': encoder_b64
                }
        
        return serialized
    
    def clear_session(self, session_id: str):
        """Clear encoders for a session."""
        if session_id in self.encoders:
            del self.encoders[session_id]
