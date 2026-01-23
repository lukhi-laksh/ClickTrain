from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from ..services.export_service import ExportService

router = APIRouter()
export_service = ExportService()

@router.get("/export/{session_id}")
async def export_model(session_id: str):
    """
    Export the trained model as a .pkl file for download.
    """
    try:
        file_path = export_service.export_model(session_id)
        return FileResponse(
            path=file_path,
            filename="trained_model.pkl",
            media_type='application/octet-stream'
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")