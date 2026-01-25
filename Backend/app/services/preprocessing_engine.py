"""
PreprocessingEngine: Main orchestrator for all preprocessing operations.
Coordinates all preprocessing services and manages dataset versions.
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
from .audit_logger import AuditLogger, ActionType


class PreprocessingEngine:
    """
    Main preprocessing orchestrator.
    Coordinates all preprocessing operations with version control and audit logging.
    """
    
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.null_handler = NullValueHandler()
        self.duplicate_handler = DuplicateHandler()
        self.constant_detector = ConstantColumnDetector()
        self.encoder_manager = EncoderManager()
        self.scaler_manager = ScalerManager()
        self.outlier_handler = OutlierHandler()
        self.sampling_handler = SamplingHandler()
        self.audit_logger = AuditLogger()
    
    def initialize_dataset(self, session_id: str, df: pd.DataFrame, dataset_name: str = None):
        """Initialize a new dataset session."""
        self.dataset_manager.initialize_session(session_id, df, dataset_name)
        self.audit_logger.log_action(
            session_id,
            ActionType.RESET,
            f"Dataset initialized: {df.shape[0]} rows × {df.shape[1]} columns",
            {'shape': df.shape, 'columns': list(df.columns)}
        )
    
    def get_dataset_stats(self, session_id: str) -> Dict:
        """Get current dataset statistics."""
        stats = self.dataset_manager.get_stats(session_id)
        action_summary = self.audit_logger.get_action_summary(session_id)
        stats['action_summary'] = action_summary
        return stats
    
    def get_action_history(self, session_id: str) -> List[Dict]:
        """Get action history for a session."""
        return self.audit_logger.get_logs(session_id)
    
    # ==================== Missing Values ====================
    
    def analyze_missing_values(self, session_id: str) -> Dict:
        """Analyze missing values in the current dataset."""
        df = self.dataset_manager.get_current(session_id)
        return self.null_handler.detect_null_values(df)
    
    def handle_missing_values(
        self,
        session_id: str,
        columns: Optional[List[str]] = None,
        strategy: str = 'mean',
        constant_value: Optional[float] = None,
        constant_string: Optional[str] = None
    ) -> Dict:
        """Handle missing values and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.null_handler.handle_missing_values(
            df, columns, strategy, constant_value, constant_string
        )
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'missing_values',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.MISSING_VALUES,
            f"Handled missing values using {strategy}",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    # ==================== Duplicates ====================
    
    def analyze_duplicates(self, session_id: str) -> Dict:
        """Analyze duplicates in the current dataset."""
        df = self.dataset_manager.get_current(session_id)
        return self.duplicate_handler.detect_duplicates(df)
    
    def remove_duplicates(
        self,
        session_id: str,
        keep: str = 'first',
        subset: Optional[List[str]] = None
    ) -> Dict:
        """Remove duplicates and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.duplicate_handler.remove_duplicates(df, keep, subset)
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'remove_duplicates',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.DUPLICATES,
            f"Removed duplicate rows (keep={keep})",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    # ==================== Constant Columns ====================
    
    def detect_constant_columns(self, session_id: str) -> Dict:
        """Detect constant columns in the current dataset."""
        df = self.dataset_manager.get_current(session_id)
        return self.constant_detector.detect_constant_columns(df)
    
    def remove_constant_columns(self, session_id: str, columns: List[str]) -> Dict:
        """Remove constant columns and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.constant_detector.remove_columns(df, columns)
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'remove_constant_columns',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.CONSTANT_COLUMNS,
            f"Removed {len(columns)} constant column(s)",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    # ==================== Encoding ====================
    
    def label_encode(self, session_id: str, columns: List[str]) -> Dict:
        """Apply Label Encoding and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.encoder_manager.label_encode(df, columns, session_id)
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'label_encoding',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.ENCODING,
            f"Applied Label Encoding to {len(columns)} column(s)",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    def one_hot_encode(
        self,
        session_id: str,
        columns: List[str],
        drop_first: bool = False,
        handle_binary: bool = True
    ) -> Dict:
        """Apply One-Hot Encoding and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.encoder_manager.one_hot_encode(
            df, columns, drop_first, handle_binary, session_id
        )
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'one_hot_encoding',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.ENCODING,
            f"Applied One-Hot Encoding to {len(columns)} column(s)",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    def ordinal_encode(
        self,
        session_id: str,
        column: str,
        categories: Optional[List] = None,
        auto_order: bool = True
    ) -> Dict:
        """Apply Ordinal Encoding and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.encoder_manager.ordinal_encode(
            df, column, categories, auto_order, session_id
        )
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'ordinal_encoding',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.ENCODING,
            f"Applied Ordinal Encoding to {column}",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    def target_encode(
        self,
        session_id: str,
        columns: List[str],
        target_column: str
    ) -> Dict:
        """Apply Target Encoding and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.encoder_manager.target_encode(
            df, columns, target_column, session_id
        )
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'target_encoding',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.ENCODING,
            f"Applied Target Encoding to {len(columns)} column(s)",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    # ==================== Scaling ====================
    
    def scale_features(
        self,
        session_id: str,
        columns: List[str],
        method: str = 'standard'
    ) -> Dict:
        """Apply feature scaling and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.scaler_manager.scale_features(
            df, columns, method, session_id
        )
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'feature_scaling',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.SCALING,
            f"Applied {method} scaling to {len(columns)} column(s)",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    # ==================== Outliers ====================
    
    def detect_outliers(
        self,
        session_id: str,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Dict:
        """Detect outliers in specified columns."""
        df = self.dataset_manager.get_current(session_id)
        return self.outlier_handler.detect_outliers(df, columns, method, threshold)
    
    def handle_outliers(
        self,
        session_id: str,
        columns: List[str],
        method: str = 'iqr',
        action: str = 'remove',
        threshold: float = 3.0
    ) -> Dict:
        """Handle outliers and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        df_processed, metadata = self.outlier_handler.handle_outliers(
            df, columns, method, action, threshold
        )
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'outlier_handling',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.OUTLIERS,
            f"Handled outliers using {method} method ({action})",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    # ==================== Sampling ====================
    
    def analyze_class_distribution(self, session_id: str, target_column: str) -> Dict:
        """Analyze class distribution."""
        df = self.dataset_manager.get_current(session_id)
        return self.sampling_handler.analyze_class_distribution(df, target_column)
    
    def apply_sampling(
        self,
        session_id: str,
        target_column: str,
        method: str = 'smote'
    ) -> Dict:
        """Apply sampling and create a new version."""
        df = self.dataset_manager.get_current(session_id)
        
        if method == 'smote':
            df_processed, metadata = self.sampling_handler.apply_smote(df, target_column)
        elif method == 'over':
            df_processed, metadata = self.sampling_handler.apply_random_oversampling(df, target_column)
        elif method == 'under':
            df_processed, metadata = self.sampling_handler.apply_random_undersampling(df, target_column)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Create new version
        version_id = self.dataset_manager.create_version(
            session_id,
            df_processed,
            'sampling',
            metadata
        )
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.SAMPLING,
            f"Applied {method} sampling",
            metadata
        )
        
        return {
            'version_id': version_id,
            'shape': df_processed.shape,
            'metadata': metadata
        }
    
    # ==================== Version Control ====================
    
    def undo(self, session_id: str) -> Optional[Dict]:
        """Undo the last action."""
        version = self.dataset_manager.undo(session_id)
        
        if version:
            self.audit_logger.log_action(
                session_id,
                ActionType.UNDO,
                f"Undid action: {version.action}",
                {'version_id': version.version_id, 'action': version.action}
            )
            return {
                'version_id': version.version_id,
                'action': version.action,
                'shape': version.shape
            }
        return None
    
    def redo(self, session_id: str) -> Optional[Dict]:
        """Redo the last undone action."""
        version = self.dataset_manager.redo(session_id)
        
        if version:
            self.audit_logger.log_action(
                session_id,
                ActionType.REDO,
                f"Redid action: {version.action}",
                {'version_id': version.version_id, 'action': version.action}
            )
            return {
                'version_id': version.version_id,
                'action': version.action,
                'shape': version.shape
            }
        return None
    
    def reset_to_original(self, session_id: str) -> Dict:
        """Reset dataset to original state."""
        version = self.dataset_manager.reset_to_original(session_id)
        
        # Clear encoders and scalers
        self.encoder_manager.clear_session(session_id)
        self.scaler_manager.clear_session(session_id)
        
        # Log action
        self.audit_logger.log_action(
            session_id,
            ActionType.RESET,
            "Reset dataset to original state",
            {'version_id': version.version_id}
        )
        
        return {
            'version_id': version.version_id,
            'shape': version.shape
        }
    
    # ==================== Export ====================
    
    def get_preprocessing_summary(self, session_id: str) -> Dict:
        """Get comprehensive preprocessing summary."""
        stats = self.get_dataset_stats(session_id)
        action_history = self.get_action_history(session_id)
        
        # Get encoders and scalers
        encoders = self.encoder_manager.serialize_encoders(session_id)
        scalers = self.scaler_manager.serialize_scalers(session_id)
        
        return {
            'dataset_stats': stats,
            'action_history': action_history,
            'encoders': encoders,
            'scalers': scalers,
            'preprocessing_complete': len(action_history) > 0
        }
    
    def get_current_dataset(self, session_id: str) -> pd.DataFrame:
        """Get the current processed dataset."""
        return self.dataset_manager.get_current(session_id)
    
    def get_encoders(self, session_id: str) -> Dict:
        """Get encoders for a session."""
        return self.encoder_manager.get_encoders(session_id)
    
    def get_scalers(self, session_id: str) -> Dict:
        """Get scalers for a session."""
        return self.scaler_manager.get_scalers(session_id)
    
    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        self.dataset_manager.clear_session(session_id)
        self.encoder_manager.clear_session(session_id)
        self.scaler_manager.clear_session(session_id)
        self.audit_logger.clear_session(session_id)
