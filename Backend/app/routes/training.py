from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from ..services.training_service import TrainingService

router = APIRouter()
training_service = TrainingService()

class TrainingConfig(BaseModel):
    algorithm: str  # linear_regression, logistic_regression, random_forest
    target_column: str

@router.post("/training/{session_id}")
async def train_model(session_id: str, config: TrainingConfig):
    """
    Train a machine learning model on the preprocessed data.
    """
    try:
        result = training_service.train_model(session_id, config.algorithm, config.target_column)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")