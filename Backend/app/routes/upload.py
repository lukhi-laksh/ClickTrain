from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import os
import uuid
from ..services.data_service import DataService
from ..services.preprocessing_engine import PreprocessingEngine

router = APIRouter()
data_service = DataService()
preprocessing_engine = PreprocessingEngine()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset file.
    Validates file type and saves temporarily.
    Returns a session ID for subsequent operations.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        # Read the file content
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))

        # Ensure all column names are strings so JSON keys and frontend indexing match
        df.columns = df.columns.map(str)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Store the dataframe in memory (full dataset with stringified column names)
        data_service.store_data(session_id, df)
        
        # Initialize preprocessing engine with the dataset
        preprocessing_engine.initialize_dataset(session_id, df, file.filename)

        # Build a lightweight preview (first 5 rows) for the frontend
        head_df = df.head(5)
        preview_records = head_df.to_dict(orient="records")

        return {
            "session_id": session_id,
            "message": "Dataset uploaded successfully",
            "shape": df.shape,
            "columns": list(df.columns),
            "preview": preview_records,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")