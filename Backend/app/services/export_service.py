import pickle
import os
from .data_service import DataService

class ExportService:
    """
    Service to export trained models as pickle files.
    """
    def __init__(self):
        self.data_service = DataService()

    def export_model(self, session_id: str) -> str:
        """
        Export the trained model as a .pkl file.
        Returns the file path for download.
        """
        processed_data = self.data_service.get_processed_data(session_id)

        if 'model' not in processed_data:
            raise ValueError("No trained model found. Please train a model first.")

        model = processed_data['model']

        # Create temp directory for model
        model_dir = f"temp_models_{session_id}"
        os.makedirs(model_dir, exist_ok=True)

        # Save model as pickle
        model_path = os.path.join(model_dir, "trained_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        return model_path