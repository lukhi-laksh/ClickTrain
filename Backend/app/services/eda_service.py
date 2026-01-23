import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .data_service import DataService

class EDAService:
    """
    Service to perform Exploratory Data Analysis on datasets.
    Returns JSON data for Plotly visualizations.
    """
    def __init__(self):
        self.data_service = DataService()

    def perform_eda(self, session_id: str) -> Dict:
        """
        Perform comprehensive EDA on the dataset.
        Returns statistics for frontend visualization.
        """
        df = self.data_service.get_data(session_id)

        # Get column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Basic statistics
        stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        }

        # Numerical statistics with skewness and kurtosis
        if len(numerical_cols) > 0:
            numerical_stats = self._get_numerical_stats(df[numerical_cols])
            stats["numerical_stats"] = numerical_stats

        # Categorical statistics
        if len(categorical_cols) > 0:
            categorical_stats = self._get_categorical_stats(df[categorical_cols])
            stats["categorical_stats"] = categorical_stats

        # Correlation matrix
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            stats["correlation_matrix"] = corr_matrix.to_dict()
            stats["top_correlations"] = self._get_top_correlations(corr_matrix)

        return {
            "statistics": stats
        }

    def _get_numerical_stats(self, df: pd.DataFrame) -> List[Dict]:
        """Get numerical statistics including skewness and kurtosis"""
        stats_list = []
        describe_df = df.describe()
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_list.append({
                    "column": col,
                    "count": float(describe_df.loc['count', col]),
                    "mean": float(describe_df.loc['mean', col]),
                    "median": float(col_data.median()),
                    "std": float(describe_df.loc['std', col]),
                    "min": float(describe_df.loc['min', col]),
                    "max": float(describe_df.loc['max', col]),
                    "q25": float(describe_df.loc['25%', col]),
                    "q50": float(describe_df.loc['50%', col]),
                    "q75": float(describe_df.loc['75%', col]),
                    "skewness": float(col_data.skew()) if len(col_data) > 2 else 0.0,
                    "kurtosis": float(col_data.kurtosis()) if len(col_data) > 2 else 0.0
                })
        
        return stats_list

    def _get_categorical_stats(self, df: pd.DataFrame) -> List[Dict]:
        """Get categorical statistics"""
        stats_list = []
        
        for col in df.columns:
            col_data = df[col]
            value_counts = col_data.value_counts()
            total_count = len(col_data)
            non_null_count = col_data.notna().sum()
            missing_count = col_data.isna().sum()
            missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
            
            unique_count = col_data.nunique()
            
            # Top category
            top_category = value_counts.index[0] if len(value_counts) > 0 else None
            top_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
            top_freq_pct = (top_freq / non_null_count * 100) if non_null_count > 0 else 0
            
            # Low frequency categories (appearing < 1% of non-null data)
            low_freq_threshold = non_null_count * 0.01
            low_freq_count = (value_counts < low_freq_threshold).sum()
            low_freq_pct = (low_freq_count / unique_count * 100) if unique_count > 0 else 0
            
            # Cardinality classification
            if unique_count <= 5:
                cardinality = "low"
            elif unique_count <= 20:
                cardinality = "medium"
            else:
                cardinality = "high"
            
            stats_list.append({
                "column": col,
                "unique_values": int(unique_count),
                "top_category": str(top_category) if top_category is not None else "N/A",
                "top_frequency": int(top_freq),
                "top_frequency_pct": round(top_freq_pct, 2),
                "low_frequency_pct": round(low_freq_pct, 2),
                "missing_count": int(missing_count),
                "missing_pct": round(missing_pct, 2),
                "cardinality": cardinality
            })
        
        return stats_list

    def _get_top_correlations(self, corr_matrix: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Get top N strongest positive correlations"""
        # Get upper triangle (avoid duplicates and diagonal)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    corr_pairs.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": float(corr_value)
                    })
        
        # Sort by correlation value (descending) and get top N
        corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
        return corr_pairs[:top_n]

    def get_column_data(self, session_id: str, column: str) -> Dict:
        """Get data for a specific column for plotting"""
        df = self.data_service.get_data(session_id)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        col_data = df[column].dropna()
        return {
            "column": column,
            "data": col_data.tolist(),
            "dtype": str(df[column].dtype)
        }

    def get_bivariate_data(self, session_id: str, x_col: str, y_col: str, hue_col: Optional[str] = None) -> Dict:
        """Get data for bivariate/multivariate plotting"""
        df = self.data_service.get_data(session_id)
        
        result = {
            "x_column": x_col,
            "y_column": y_col,
            "x_data": df[x_col].dropna().tolist(),
            "y_data": df[y_col].dropna().tolist(),
            "x_dtype": str(df[x_col].dtype),
            "y_dtype": str(df[y_col].dtype)
        }
        
        if hue_col and hue_col in df.columns:
            result["hue_column"] = hue_col
            result["hue_data"] = df[hue_col].tolist()
            result["hue_dtype"] = str(df[hue_col].dtype)
        
        return result