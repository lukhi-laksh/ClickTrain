import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
from .data_service import DataService

class TrainingService:
    """
    Service to train machine learning models and calculate performance metrics.
    """
    def __init__(self):
        self.data_service = DataService()

    def train_model(self, session_id: str, algorithm: str, target_column: str) -> Dict:
        """
        Train a model on the preprocessed data.
        Returns model performance metrics.
        """
        processed_data = self.data_service.get_processed_data(session_id)
        df = processed_data['data']

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select and train model
        model = self._get_model(algorithm, y)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, algorithm)

        # Store model in processed data
        processed_data['model'] = model
        processed_data['algorithm'] = algorithm
        processed_data['target_column'] = target_column
        self.data_service.store_processed_data(session_id, processed_data)

        return {
            "algorithm": algorithm,
            "metrics": metrics,
            "message": "Model trained successfully"
        }

    def _get_model(self, algorithm: str, y):
        """Get the appropriate model based on algorithm and target type"""
        is_classification = len(np.unique(y)) < 20  # Simple heuristic

        if algorithm == 'linear_regression':
            return LinearRegression()
        elif algorithm == 'logistic_regression':
            return LogisticRegression(random_state=42)
        elif algorithm == 'random_forest':
            if is_classification:
                return RandomForestClassifier(random_state=42)
            else:
                return RandomForestRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _calculate_metrics(self, y_true, y_pred, algorithm: str) -> Dict:
        """Calculate appropriate metrics based on problem type"""
        is_classification = algorithm in ['logistic_regression', 'random_forest'] and len(np.unique(y_true)) < 20

        if is_classification:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted')),
                "recall": float(recall_score(y_true, y_pred, average='weighted')),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted'))
            }
        else:
            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2_score": float(r2_score(y_true, y_pred))
            }