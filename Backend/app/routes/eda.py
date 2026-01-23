from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.eda_service import EDAService
from ..services.data_service import DataService

router = APIRouter()
eda_service = EDAService()
data_service = DataService()  # Get the singleton instance

@router.get("/eda/{session_id}")
async def perform_eda(session_id: str):
    """
    Perform Exploratory Data Analysis on the uploaded dataset.
    Returns statistics for frontend visualization.
    """
    try:
        # Debug: Check if session exists
        if session_id not in data_service.data_store:
            available_sessions = list(data_service.data_store.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Available sessions: {available_sessions[:5] if available_sessions else 'none'}. Please upload a dataset first."
            )
        
        result = eda_service.perform_eda(session_id)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Session not found: {session_id}. Please upload a dataset first."
        )
    except Exception as e:
        import traceback
        error_detail = f"EDA failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/eda/{session_id}/columns")
async def get_columns(session_id: str):
    """Get list of all columns with their types"""
    try:
        result = eda_service.perform_eda(session_id)
        return {
            "all_columns": result["statistics"]["columns"],
            "numerical_columns": result["statistics"].get("numerical_columns", []),
            "categorical_columns": result["statistics"].get("categorical_columns", [])
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