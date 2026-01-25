"""
AuditLogger: Tracks all preprocessing actions for audit and reproducibility.
"""
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class ActionType(Enum):
    """Types of preprocessing actions."""
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    CONSTANT_COLUMNS = "constant_columns"
    ENCODING = "encoding"
    SCALING = "scaling"
    OUTLIERS = "outliers"
    SAMPLING = "sampling"
    RESET = "reset"
    UNDO = "undo"
    REDO = "redo"


class AuditLogger:
    """
    Logs all preprocessing actions for audit trail and reproducibility.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuditLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not AuditLogger._initialized:
            # Store logs per session: {session_id: [LogEntry]}
            self.logs: Dict[str, List[Dict]] = {}
            AuditLogger._initialized = True
    
    def log_action(
        self,
        session_id: str,
        action_type: ActionType,
        description: str,
        metadata: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Log a preprocessing action.
        
        Args:
            session_id: Session identifier
            action_type: Type of action
            description: Human-readable description
            metadata: Additional metadata about the action
            success: Whether the action succeeded
            error_message: Error message if action failed
        """
        if session_id not in self.logs:
            self.logs[session_id] = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type.value,
            'description': description,
            'metadata': metadata or {},
            'success': success,
            'error_message': error_message
        }
        
        self.logs[session_id].append(log_entry)
    
    def get_logs(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get audit logs for a session.
        
        Args:
            limit: Maximum number of logs to return (None = all)
        
        Returns:
            List of log entries
        """
        if session_id not in self.logs:
            return []
        
        logs = self.logs[session_id]
        if limit:
            return logs[-limit:]
        return logs
    
    def get_action_summary(self, session_id: str) -> Dict:
        """
        Get summary of actions for a session.
        
        Returns:
            Dictionary with action counts and summary
        """
        if session_id not in self.logs:
            return {
                'total_actions': 0,
                'successful_actions': 0,
                'failed_actions': 0,
                'action_types': {}
            }
        
        logs = self.logs[session_id]
        
        summary = {
            'total_actions': len(logs),
            'successful_actions': sum(1 for log in logs if log['success']),
            'failed_actions': sum(1 for log in logs if not log['success']),
            'action_types': {}
        }
        
        # Count by action type
        for log in logs:
            action_type = log['action_type']
            if action_type not in summary['action_types']:
                summary['action_types'][action_type] = 0
            summary['action_types'][action_type] += 1
        
        return summary
    
    def clear_session(self, session_id: str):
        """Clear logs for a session."""
        if session_id in self.logs:
            del self.logs[session_id]
