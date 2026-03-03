"""
EDA Routes
All responses are serialised via _safe_json_response() which converts every
numpy type and NaN/Inf value to a JSON-compatible Python type BEFORE handing
the bytes to FastAPI — so Python's standard json.dumps never sees a numpy
scalar or a NaN float.
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from typing import Optional
import json
import math
import numpy as np
from ..services.eda_service import EDAService
from ..services.data_service import DataService
from ..services.dataset_manager import DatasetManager

router = APIRouter()
eda_service = EDAService()
data_service = DataService()
_dm = DatasetManager()


# ── JSON-safe serialisation ────────────────────────────────────────────────────

def _to_safe(obj):
    """
    Recursively convert any Python / NumPy object tree to a structure that
    Python's json.dumps can serialise without errors:
      • NaN / ±Inf float  → None  (becomes JSON null)
      • numpy integer     → int
      • numpy float       → float (or None if NaN/Inf)
      • numpy bool        → bool
      • numpy ndarray     → list (then recursed)
      • dict / list       → recursed
      • everything else   → unchanged (str, int, float, bool, None are fine)
    """
    # ── containers ──────────────────────────────────────────────────────────
    if isinstance(obj, dict):
        return {str(k): _to_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_safe(v) for v in obj]

    # ── numpy types — MUST come before plain Python checks because
    #    np.float64 is a subclass of float in NumPy < 2.0 ──────────────────
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):          # np.int8/16/32/64/…
        return int(obj)
    if isinstance(obj, np.floating):         # np.float32/64/…
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [_to_safe(v) for v in obj.tolist()]

    # ── plain Python float (also catches np.float64 in older NumPy) ─────────
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # ── plain Python int / bool / str / None — already JSON-safe ─────────────
    return obj


def _safe_json_response(data) -> Response:
    """
    Convert `data` to JSON bytes with guaranteed NaN/Inf safety, then return
    a raw Response so FastAPI never runs its own JSON serializer on the data.
    """
    safe = _to_safe(data)
    try:
        # allow_nan=True is fine here because _to_safe already replaced every
        # NaN/Inf with None — there's nothing left to produce NaN tokens.
        body = json.dumps(safe, allow_nan=True)
    except Exception:
        # Absolute last resort: stringify any remaining unserializable types
        body = json.dumps(safe, allow_nan=True, default=str)
    return Response(content=body, media_type="application/json")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/eda/{session_id}")
async def perform_eda(session_id: str):
    """
    Perform Exploratory Data Analysis on the uploaded dataset.
    Returns statistics for frontend visualisation.
    Reads from the live in-memory DatasetManager so preprocessing changes are reflected.
    """
    try:
        try:
            df = _dm.get_current(session_id)
        except ValueError:
            if session_id not in data_service.data_store:
                available = list(data_service.data_store.keys())
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Session '{session_id}' not found. "
                        f"Available: {available[:5] if available else 'none'}. "
                        "Please upload first."
                    ),
                )
            df = data_service.get_data(session_id)

        numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Build stats — values may still contain numpy types / NaN here;
        # _safe_json_response handles all of that.
        stats = {
            "shape":               list(df.shape),
            "columns":             list(df.columns),
            "numerical_columns":   numerical_cols,
            "categorical_columns": categorical_cols,
            "dtypes":              df.dtypes.astype(str).to_dict(),
            "missing_values":      df.isnull().sum().to_dict(),
            "missing_percentage":  (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        }

        if numerical_cols:
            stats["numerical_stats"] = eda_service._get_numerical_stats(df[numerical_cols])

        if categorical_cols:
            stats["categorical_stats"] = eda_service._get_categorical_stats(df[categorical_cols])

        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            stats["correlation_matrix"] = corr_matrix.to_dict()
            stats["top_correlations"]   = eda_service._get_top_correlations(corr_matrix)

        return _safe_json_response({"statistics": stats})

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}. Please upload first.")
    except Exception as e:
        import traceback, sys
        tb = traceback.format_exc()
        print(f"\n\n===EDA ERROR===\n{tb}\n===============", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}\nTraceback:\n{tb}")


@router.get("/eda/{session_id}/fast-columns")
async def get_fast_columns(session_id: str):
    """
    Ultra-fast endpoint: returns only column names and types.
    Used by the preprocessing page on load — avoids running full EDA.
    """
    try:
        try:
            df = _dm.get_current(session_id)
        except ValueError:
            df = data_service.get_data(session_id)

        numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        return _safe_json_response({
            "statistics": {
                "columns":             list(df.columns),
                "numerical_columns":   numerical_cols,
                "categorical_columns": categorical_cols,
            }
        })
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get columns: {str(e)}")


@router.get("/eda/{session_id}/columns")
async def get_columns(session_id: str):
    """Get list of all columns with their types."""
    try:
        try:
            df = _dm.get_current(session_id)
        except ValueError:
            df = data_service.get_data(session_id)

        numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        return _safe_json_response({
            "all_columns":         list(df.columns),
            "numerical_columns":   numerical_cols,
            "categorical_columns": categorical_cols,
        })
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get columns: {str(e)}")


@router.get("/eda/{session_id}/plot-data")
async def get_plot_data(
    session_id: str,
    x_col: Optional[str] = Query(None),
    y_col: Optional[str] = Query(None),
    hue_col: Optional[str] = Query(None),
):
    """Get data for custom plotting"""
    try:
        if x_col and y_col:
            data = eda_service.get_bivariate_data(session_id, x_col, y_col, hue_col)
        elif x_col:
            data = eda_service.get_column_data(session_id, x_col)
        else:
            raise HTTPException(status_code=400, detail="At least x_col must be provided")

        return _safe_json_response(data)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plot data: {str(e)}")