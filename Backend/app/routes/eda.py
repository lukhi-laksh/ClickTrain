from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import numpy as np
from ..services.eda_service import EDAService
from ..services.data_service import DataService
from ..services.dataset_manager import DatasetManager

router = APIRouter()
eda_service = EDAService()
data_service = DataService()  # Get the singleton instance
_dm = DatasetManager()        # In-memory dataset manager (same singleton used by preprocessing)

@router.get("/eda/{session_id}")
async def perform_eda(session_id: str):
    """
    Perform Exploratory Data Analysis on the uploaded dataset.
    Returns statistics for frontend visualization.
    Reads from the live in-memory DatasetManager so preprocessing changes are reflected.
    """
    try:
        # Prefer the live preprocessed DataFrame from DatasetManager
        try:
            df = _dm.get_current(session_id)
        except ValueError:
            # Fallback: try DataService (original upload store)
            if session_id not in data_service.data_store:
                available_sessions = list(data_service.data_store.keys())
                raise HTTPException(
                    status_code=404,
                    detail=f"Session '{session_id}' not found. Available: {available_sessions[:5] if available_sessions else 'none'}. Please upload first."
                )
            df = data_service.get_data(session_id)

        # Fast column-type extraction using pandas dtype system
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        stats = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        }

        # Numerical stats — only when at least one numeric col
        if numerical_cols:
            stats["numerical_stats"] = eda_service._get_numerical_stats(df[numerical_cols])

        # Categorical stats
        if categorical_cols:
            stats["categorical_stats"] = eda_service._get_categorical_stats(df[categorical_cols])

        # Correlation — only when multiple numeric cols
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            stats["correlation_matrix"] = corr_matrix.to_dict()
            stats["top_correlations"] = eda_service._get_top_correlations(corr_matrix)

        return {"statistics": stats}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}. Please upload first.")
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}\n{traceback.format_exc()}")


@router.get("/eda/{session_id}/fast-columns")
async def get_fast_columns(session_id: str):
    """
    Ultra-fast endpoint: returns only column names and types.
    Used by the preprocessing page on load — avoids running full EDA.
    Reads directly from the live in-memory DatasetManager.
    """
    try:
        try:
            df = _dm.get_current(session_id)
        except ValueError:
            df = data_service.get_data(session_id)

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        return {
            "statistics": {
                "columns": list(df.columns),
                "numerical_columns": numerical_cols,
                "categorical_columns": categorical_cols,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get columns: {str(e)}")


@router.get("/eda/{session_id}/columns")
async def get_columns(session_id: str):
    """Get list of all columns with their types (delegates to fast-columns)."""
    try:
        try:
            df = _dm.get_current(session_id)
        except ValueError:
            df = data_service.get_data(session_id)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        return {
            "all_columns": list(df.columns),
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get columns: {str(e)}")


@router.get("/eda/{session_id}/plot-data")
async def get_plot_data(
    session_id: str,
    x_col: Optional[str] = Query(None),
    y_col: Optional[str] = Query(None),
    hue_col: Optional[str] = Query(None)
):
    """Get data for custom plotting"""
    try:
        if x_col and y_col:
            return eda_service.get_bivariate_data(session_id, x_col, y_col, hue_col)
        elif x_col:
            return eda_service.get_column_data(session_id, x_col)
        else:
            raise HTTPException(status_code=400, detail="At least x_col must be provided")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plot data: {str(e)}")