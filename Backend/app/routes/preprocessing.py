"""
Preprocessing API Routes — grandmaster edition.

New endpoints added:
  GET  /preprocessing/{sid}/column-registry   → column roles at a glance
  GET  /preprocessing/{sid}/scalable-columns  → safe-to-scale column list
  GET  /preprocessing/{sid}/encodable-columns → columns still needing encoding

Updated request models:
  - MissingValuesRequest: +min_valid_ratio, +strategy 'knn'/'smart'/'ffill'/'bfill'
  - OneHotEncodingRequest: +max_categories guard
  - TargetEncodingRequest: +smoothing parameter
  - ScalingRequest: now returns skipped_columns with explanations
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
    strategy: str = Field(
        ...,
        description=(
            "Strategy: drop | mean | median | mode | ffill | bfill | "
            "knn | smart | constant_num | constant_cat"
        )
    )
    constant_value:   Optional[float] = None
    constant_string:  Optional[str]   = None
    min_valid_ratio:  float = Field(
        0.0,
        description=(
            "For 'drop' strategy: minimum fraction of non-null values a row "
            "must have across selected columns to be kept (0.0 = drop any null row)."
        )
    )

class DuplicateRequest(BaseModel):
    keep:   str = Field("first", description="Keep: first | last")
    subset: Optional[List[str]] = None

class ConstantColumnsRequest(BaseModel):
    columns: List[str]

class LabelEncodingRequest(BaseModel):
    columns: List[str]

class OneHotEncodingRequest(BaseModel):
    columns:         List[str]
    drop_first:      bool = False
    handle_binary:   bool = True
    max_categories:  int  = Field(
        50,
        description="Refuse OHE for columns with more than this many unique values."
    )

class OrdinalEncodingRequest(BaseModel):
    column:     str
    categories: Optional[List[str]] = None
    auto_order: bool = True

class TargetEncodingRequest(BaseModel):
    columns:       List[str]
    target_column: str
    smoothing:     float = Field(
        10.0,
        description=(
            "Smoothing factor for Bayesian target encoding. "
            "Higher = more regularisation = less leakage."
        )
    )

class ScalingRequest(BaseModel):
    columns: List[str]
    method:  str = Field(
        "standard",
        description="Method: standard | minmax | robust"
    )

class OutlierDetectionRequest(BaseModel):
    columns:   List[str]
    method:    str   = Field("iqr", description="Method: iqr | zscore")
    threshold: float = 3.0

class OutlierHandlingRequest(BaseModel):
    columns:   List[str]
    method:    str   = Field("iqr",    description="Method: iqr | zscore")
    action:    str   = Field("remove", description="Action: remove | cap | flag")
    threshold: float = 3.0

class SamplingRequest(BaseModel):
    target_column: str
    method:        str = Field("smote", description="Method: smote | over | under")


# ==================== Missing Values ====================

@router.get("/preprocessing/{session_id}/missing-values")
async def get_missing_values_analysis(session_id: str):
    """Analyse missing values across the current dataset."""
    try:
        return preprocessing_engine.analyze_missing_values(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to analyse missing values: {e}")


@router.post("/preprocessing/{session_id}/missing-values")
async def handle_missing_values(session_id: str, req: MissingValuesRequest):
    """Handle missing values. Strategy-aware and registry-safe."""
    try:
        return preprocessing_engine.handle_missing_values(
            session_id,
            req.columns,
            req.strategy,
            req.constant_value,
            req.constant_string,
            req.min_valid_ratio,
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to handle missing values: {e}")


# ==================== Duplicates ====================

@router.get("/preprocessing/{session_id}/duplicates")
async def get_duplicates_analysis(session_id: str):
    try:
        return preprocessing_engine.analyze_duplicates(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to analyse duplicates: {e}")


@router.post("/preprocessing/{session_id}/duplicates")
async def remove_duplicates(session_id: str, req: DuplicateRequest):
    try:
        return preprocessing_engine.remove_duplicates(session_id, req.keep, req.subset)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to remove duplicates: {e}")


# ==================== Constant Columns ====================

@router.get("/preprocessing/{session_id}/constant-columns")
async def get_constant_columns(session_id: str):
    try:
        return preprocessing_engine.detect_constant_columns(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to detect constant columns: {e}")


@router.post("/preprocessing/{session_id}/constant-columns")
async def remove_constant_columns(session_id: str, req: ConstantColumnsRequest):
    try:
        return preprocessing_engine.remove_constant_columns(session_id, req.columns)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to remove constant columns: {e}")


# ==================== Encoding ====================

@router.post("/preprocessing/{session_id}/encoding/label")
async def apply_label_encoding(session_id: str, req: LabelEncodingRequest):
    """
    Apply Label Encoding.
    Already-encoded or numeric columns are skipped (see metadata.columns_skipped).
    """
    try:
        return preprocessing_engine.label_encode(session_id, req.columns)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to apply label encoding: {e}")


@router.post("/preprocessing/{session_id}/encoding/onehot")
async def apply_onehot_encoding(session_id: str, req: OneHotEncodingRequest):
    """
    Apply One-Hot Encoding.
    Binary columns (2 unique values) use deterministic 0/1 in-place encoding.
    Columns with > max_categories unique values are refused.
    """
    try:
        return preprocessing_engine.one_hot_encode(
            session_id,
            req.columns,
            req.drop_first,
            req.handle_binary,
            req.max_categories,
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to apply one-hot encoding: {e}")


@router.post("/preprocessing/{session_id}/encoding/ordinal")
async def apply_ordinal_encoding(session_id: str, req: OrdinalEncodingRequest):
    """
    Apply Ordinal Encoding.
    Unknown categories are mapped to -1 with a warning (not silently to NaN).
    """
    try:
        return preprocessing_engine.ordinal_encode(
            session_id, req.column, req.categories, req.auto_order
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to apply ordinal encoding: {e}")


@router.post("/preprocessing/{session_id}/encoding/target")
async def apply_target_encoding(session_id: str, req: TargetEncodingRequest):
    """
    Apply Smoothed Target Encoding (Bayesian mean estimation).
    Rare categories are regularised toward the global mean.
    """
    try:
        return preprocessing_engine.target_encode(
            session_id, req.columns, req.target_column, req.smoothing
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to apply target encoding: {e}")


# ==================== Scaling ====================

@router.post("/preprocessing/{session_id}/scaling")
async def apply_scaling(session_id: str, req: ScalingRequest):
    """
    Apply feature scaling.
    Encoded columns are automatically skipped — see metadata.columns_skipped
    for the list of skipped columns and the reason.
    """
    try:
        return preprocessing_engine.scale_features(session_id, req.columns, req.method)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to apply scaling: {e}")


# ==================== Column Registry ====================

@router.get("/preprocessing/{session_id}/column-registry")
async def get_column_registry(session_id: str):
    """
    Get the column role registry.
    Shows whether each column is: original_numeric, original_categorical,
    label_encoded, ordinal_encoded, target_encoded, one_hot, binary_encoded,
    scaled, or dropped.
    """
    try:
        return preprocessing_engine.get_column_registry(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get column registry: {e}")


@router.get("/preprocessing/{session_id}/scalable-columns")
async def get_scalable_columns(session_id: str):
    """Return columns that are currently safe to scale (original_numeric and target_encoded)."""
    try:
        cols = preprocessing_engine.get_scalable_columns(session_id)
        return {'scalable_columns': cols, 'count': len(cols)}
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get scalable columns: {e}")


@router.get("/preprocessing/{session_id}/encodable-columns")
async def get_encodable_columns(session_id: str):
    """Return columns that still need encoding (original_categorical only)."""
    try:
        cols = preprocessing_engine.get_encodable_columns(session_id)
        return {'encodable_columns': cols, 'count': len(cols)}
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get encodable columns: {e}")


@router.get("/preprocessing/{session_id}/column-values")
async def get_column_unique_values(session_id: str, column: str):
    """
    Return the sorted unique non-null values for a specific column.
    Used by the Ordinal Encoding UI to populate the drag-and-drop order builder
    so users never have to manually type category names.
    """
    try:
        df = preprocessing_engine.get_current_dataset(session_id)
        if column not in df.columns:
            raise HTTPException(404, detail=f"Column '{column}' not found.")
        raw = df[column].dropna().unique().tolist()
        # Return as strings, sorted where possible
        str_vals = [str(v) for v in raw]
        try:
            str_vals = sorted(str_vals)
        except Exception:
            pass
        return {'column': column, 'values': str_vals, 'count': len(str_vals)}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get column values: {e}")


# ==================== Outliers ====================

@router.post("/preprocessing/{session_id}/outliers/detect")
async def detect_outliers(session_id: str, req: OutlierDetectionRequest):
    try:
        return preprocessing_engine.detect_outliers(
            session_id, req.columns, req.method, req.threshold
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to detect outliers: {e}")


@router.post("/preprocessing/{session_id}/outliers/handle")
async def handle_outliers(session_id: str, req: OutlierHandlingRequest):
    try:
        return preprocessing_engine.handle_outliers(
            session_id, req.columns, req.method, req.action, req.threshold
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to handle outliers: {e}")


# ==================== Sampling ====================

@router.get("/preprocessing/{session_id}/sampling/distribution")
async def get_class_distribution(session_id: str, target_column: str):
    try:
        return preprocessing_engine.analyze_class_distribution(session_id, target_column)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to analyse class distribution: {e}")


@router.post("/preprocessing/{session_id}/sampling")
async def apply_sampling(session_id: str, req: SamplingRequest):
    try:
        return preprocessing_engine.apply_sampling(
            session_id, req.target_column, req.method
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to apply sampling: {e}")


# ==================== Quick Summary (topbar badges) ====================

@router.get("/preprocessing/{session_id}/quick-summary")
async def get_quick_summary(session_id: str):
    """
    Single fast call that returns all three badge values:
      - total_nulls         (for the Null badge)
      - rows_to_remove      (duplicate rows that will actually be deleted)
      - constant_columns    (number of constant cols detected)
    """
    try:
        return preprocessing_engine.quick_summary(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get quick summary: {e}")


# ==================== Version Control ====================

@router.get("/preprocessing/{session_id}/stats")
async def get_dataset_stats(session_id: str):
    try:
        return preprocessing_engine.get_dataset_stats(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get stats: {e}")


@router.get("/preprocessing/{session_id}/history")
async def get_action_history(session_id: str):
    try:
        return preprocessing_engine.get_action_history(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get history: {e}")


@router.post("/preprocessing/{session_id}/undo")
async def undo_action(session_id: str):
    try:
        result = preprocessing_engine.undo(session_id)
        if result is None:
            raise HTTPException(400, detail="Nothing to undo")
        return result
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to undo: {e}")


@router.post("/preprocessing/{session_id}/redo")
async def redo_action(session_id: str):
    try:
        result = preprocessing_engine.redo(session_id)
        if result is None:
            raise HTTPException(400, detail="Nothing to redo")
        return result
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to redo: {e}")


@router.post("/preprocessing/{session_id}/reset")
async def reset_dataset(session_id: str):
    try:
        return preprocessing_engine.reset_to_original(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to reset: {e}")


# ==================== Summary & Export ====================

@router.get("/preprocessing/{session_id}/summary")
async def get_preprocessing_summary(session_id: str):
    try:
        return preprocessing_engine.get_preprocessing_summary(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get summary: {e}")


@router.get("/preprocessing/{session_id}/encoders")
async def get_encoders(session_id: str):
    try:
        return preprocessing_engine.get_encoders(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get encoders: {e}")


@router.get("/preprocessing/{session_id}/scalers")
async def get_scalers(session_id: str):
    try:
        return preprocessing_engine.get_scalers(session_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get scalers: {e}")


# ==================== CSV Export ====================

from fastapi.responses import StreamingResponse
import io


@router.get("/preprocessing/{session_id}/export-csv")
async def export_dataset_csv(session_id: str):
    """Export the current processed dataset as a downloadable CSV file."""
    try:
        df = preprocessing_engine.get_current_dataset(session_id)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    f"attachment; filename=preprocessed_{session_id[:8]}.csv"
            }
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to export dataset: {e}")
