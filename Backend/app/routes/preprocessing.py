from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from ..services.preprocessing_service import PreprocessingService

router = APIRouter()
preprocessing_service = PreprocessingService()

class PreprocessingConfig(BaseModel):
    missing_strategy: str = "mean"  # mean, median, mode, drop
    scaling: bool = True
    test_size: float = 0.2

@router.post("/preprocessing/{session_id}")
async def preprocess_data(session_id: str, config: PreprocessingConfig):
    """
    Preprocess the dataset: handle missing values, encode categoricals, scale features, split data.
    """
    try:
        result = preprocessing_service.preprocess_data(session_id, config.dict())
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")