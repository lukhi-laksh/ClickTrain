"""
Preprocessing API Routes
Comprehensive endpoints for all preprocessing operations.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from ..services.preprocessing_engine import PreprocessingEngine
from ..services.data_service import DataService

router = APIRouter()
preprocessing_engine = PreprocessingEngine()
data_service = DataService()

# ==================== Request Models ====================

class MissingValuesRequest(BaseModel):
    columns: Optional[List[str]] = None
    strategy: str = Field(..., description="Strategy: drop, mean, median, mode, constant_num, constant_cat")
    constant_value: Optional[float] = None
    constant_string: Optional[str] = None

class DuplicateRequest(BaseModel):
    keep: str = Field("first", description="Keep: first, last")
    subset: Optional[List[str]] = None

class ConstantColumnsRequest(BaseModel):
    columns: List[str]

class LabelEncodingRequest(BaseModel):
    columns: List[str]

class OneHotEncodingRequest(BaseModel):
    columns: List[str]
    drop_first: bool = False
    handle_binary: bool = True

class OrdinalEncodingRequest(BaseModel):
    column: str
    categories: Optional[List[str]] = None
    auto_order: bool = True

class TargetEncodingRequest(BaseModel):
    columns: List[str]
    target_column: str

class ScalingRequest(BaseModel):
    columns: List[str]
    method: str = Field("standard", description="Method: standard, minmax, robust")

class OutlierDetectionRequest(BaseModel):
    columns: List[str]
    method: str = Field("iqr", description="Method: iqr, zscore")
    threshold: float = 3.0

class OutlierHandlingRequest(BaseModel):
    columns: List[str]
    method: str = Field("iqr", description="Method: iqr, zscore")
    action: str = Field("remove", description="Action: remove, cap, flag")
    threshold: float = 3.0

class SamplingRequest(BaseModel):
    target_column: str
    method: str = Field("smote", description="Method: smote, over, under")

# ==================== Missing Values ====================

@router.get("/preprocessing/{session_id}/missing-values")
async def get_missing_values_analysis(session_id: str):
    """Get missing values analysis for the current dataset."""
    try:
        result = preprocessing_engine.analyze_missing_values(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze missing values: {str(e)}")

@router.post("/preprocessing/{session_id}/missing-values")
async def handle_missing_values(session_id: str, request: MissingValuesRequest):
    """Handle missing values according to strategy."""
    try:
        result = preprocessing_engine.handle_missing_values(
            session_id,
            request.columns,
            request.strategy,
            request.constant_value,
            request.constant_string
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to handle missing values: {str(e)}")

# ==================== Duplicates ====================

@router.get("/preprocessing/{session_id}/duplicates")
async def get_duplicates_analysis(session_id: str):
    """Get duplicate rows and columns analysis."""
    try:
        result = preprocessing_engine.analyze_duplicates(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze duplicates: {str(e)}")

@router.post("/preprocessing/{session_id}/duplicates")
async def remove_duplicates(session_id: str, request: DuplicateRequest):
    """Remove duplicate rows."""
    try:
        result = preprocessing_engine.remove_duplicates(
            session_id,
            request.keep,
            request.subset
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove duplicates: {str(e)}")

# ==================== Constant Columns ====================

@router.get("/preprocessing/{session_id}/constant-columns")
async def get_constant_columns(session_id: str):
    """Detect constant columns."""
    try:
        result = preprocessing_engine.detect_constant_columns(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect constant columns: {str(e)}")

@router.post("/preprocessing/{session_id}/constant-columns")
async def remove_constant_columns(session_id: str, request: ConstantColumnsRequest):
    """Remove constant columns."""
    try:
        result = preprocessing_engine.remove_constant_columns(session_id, request.columns)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove constant columns: {str(e)}")

# ==================== Encoding ====================

@router.post("/preprocessing/{session_id}/encoding/label")
async def apply_label_encoding(session_id: str, request: LabelEncodingRequest):
    """Apply Label Encoding to specified columns."""
    try:
        result = preprocessing_engine.label_encode(session_id, request.columns)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply label encoding: {str(e)}")

@router.post("/preprocessing/{session_id}/encoding/onehot")
async def apply_onehot_encoding(session_id: str, request: OneHotEncodingRequest):
    """Apply One-Hot Encoding to specified columns."""
    try:
        result = preprocessing_engine.one_hot_encode(
            session_id,
            request.columns,
            request.drop_first,
            request.handle_binary
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply one-hot encoding: {str(e)}")

@router.post("/preprocessing/{session_id}/encoding/ordinal")
async def apply_ordinal_encoding(session_id: str, request: OrdinalEncodingRequest):
    """Apply Ordinal Encoding to a column."""
    try:
        result = preprocessing_engine.ordinal_encode(
            session_id,
            request.column,
            request.categories,
            request.auto_order
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply ordinal encoding: {str(e)}")

@router.post("/preprocessing/{session_id}/encoding/target")
async def apply_target_encoding(session_id: str, request: TargetEncodingRequest):
    """Apply Target Encoding to specified columns."""
    try:
        result = preprocessing_engine.target_encode(
            session_id,
            request.columns,
            request.target_column
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply target encoding: {str(e)}")

# ==================== Scaling ====================

@router.post("/preprocessing/{session_id}/scaling")
async def apply_scaling(session_id: str, request: ScalingRequest):
    """Apply feature scaling to specified columns."""
    try:
        result = preprocessing_engine.scale_features(
            session_id,
            request.columns,
            request.method
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply scaling: {str(e)}")

# ==================== Outliers ====================

@router.post("/preprocessing/{session_id}/outliers/detect")
async def detect_outliers(session_id: str, request: OutlierDetectionRequest):
    """Detect outliers in specified columns."""
    try:
        result = preprocessing_engine.detect_outliers(
            session_id,
            request.columns,
            request.method,
            request.threshold
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect outliers: {str(e)}")

@router.post("/preprocessing/{session_id}/outliers/handle")
async def handle_outliers(session_id: str, request: OutlierHandlingRequest):
    """Handle outliers according to action."""
    try:
        result = preprocessing_engine.handle_outliers(
            session_id,
            request.columns,
            request.method,
            request.action,
            request.threshold
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to handle outliers: {str(e)}")

# ==================== Sampling ====================

@router.get("/preprocessing/{session_id}/sampling/distribution")
async def get_class_distribution(session_id: str, target_column: str):
    """Get class distribution for a target column."""
    try:
        result = preprocessing_engine.analyze_class_distribution(session_id, target_column)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze class distribution: {str(e)}")

@router.post("/preprocessing/{session_id}/sampling")
async def apply_sampling(session_id: str, request: SamplingRequest):
    """Apply sampling to balance classes."""
    try:
        result = preprocessing_engine.apply_sampling(
            session_id,
            request.target_column,
            request.method
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply sampling: {str(e)}")

# ==================== Version Control ====================

@router.get("/preprocessing/{session_id}/stats")
async def get_dataset_stats(session_id: str):
    """Get current dataset statistics and action summary."""
    try:
        result = preprocessing_engine.get_dataset_stats(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/preprocessing/{session_id}/history")
async def get_action_history(session_id: str):
    """Get action history for the session."""
    try:
        result = preprocessing_engine.get_action_history(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.post("/preprocessing/{session_id}/undo")
async def undo_action(session_id: str):
    """Undo the last preprocessing action."""
    try:
        result = preprocessing_engine.undo(session_id)
        if result is None:
            raise HTTPException(status_code=400, detail="Nothing to undo")
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to undo: {str(e)}")

@router.post("/preprocessing/{session_id}/redo")
async def redo_action(session_id: str):
    """Redo the last undone action."""
    try:
        result = preprocessing_engine.redo(session_id)
        if result is None:
            raise HTTPException(status_code=400, detail="Nothing to redo")
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to redo: {str(e)}")

@router.post("/preprocessing/{session_id}/reset")
async def reset_dataset(session_id: str):
    """Reset dataset to original state."""
    try:
        result = preprocessing_engine.reset_to_original(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")

# ==================== Summary & Export ====================

@router.get("/preprocessing/{session_id}/summary")
async def get_preprocessing_summary(session_id: str):
    """Get comprehensive preprocessing summary including encoders and scalers."""
    try:
        result = preprocessing_engine.get_preprocessing_summary(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@router.get("/preprocessing/{session_id}/encoders")
async def get_encoders(session_id: str):
    """Get all encoders for the session."""
    try:
        result = preprocessing_engine.get_encoders(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get encoders: {str(e)}")

@router.get("/preprocessing/{session_id}/scalers")
async def get_scalers(session_id: str):
    """Get all scalers for the session."""
    try:
        result = preprocessing_engine.get_scalers(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scalers: {str(e)}")
