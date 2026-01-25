"""
SamplingHandler: Handles class imbalance using SMOTE, oversampling, and undersampling.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class SamplingHandler:
    """
    Handles class imbalance through various sampling techniques.
    
    IMPORTANT: Sampling should only be applied to training data after train-test split.
    This handler assumes the data provided is training data only.
    """
    
    def __init__(self):
        pass
    
    def analyze_class_distribution(self, df: pd.DataFrame, target_column: str) -> Dict:
        """
        Analyze class distribution in the dataset.
        
        Returns:
            Dictionary with class distribution statistics
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found.")
        
        class_counts = df[target_column].value_counts().to_dict()
        total_samples = len(df)
        
        class_distribution = []
        for class_val, count in class_counts.items():
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            class_distribution.append({
                'class': str(class_val),
                'count': int(count),
                'percentage': round(percentage, 2)
            })
        
        # Calculate imbalance ratio
        if len(class_counts) > 1:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 1.0
        
        return {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_distribution': class_distribution,
            'imbalance_ratio': round(imbalance_ratio, 2),
            'is_balanced': imbalance_ratio <= 2.0  # Consider balanced if ratio <= 2:1
        }
    
    def apply_smote(
        self,
        df: pd.DataFrame,
        target_column: str,
        k_neighbors: int = 5
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique).
        
        Note: Requires imbalanced-learn library.
        Falls back to random oversampling if not available.
        
        Returns:
            Tuple of (sampled_df, metadata)
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            # Fallback to random oversampling
            return self.apply_random_oversampling(df, target_column)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Apply SMOTE
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Reconstruct dataframe
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled
        
        # Get before/after stats
        before_dist = self.analyze_class_distribution(df, target_column)
        after_dist = self.analyze_class_distribution(df_resampled, target_column)
        
        metadata = {
            'method': 'smote',
            'k_neighbors': k_neighbors,
            'before_distribution': before_dist,
            'after_distribution': after_dist,
            'samples_added': len(df_resampled) - len(df)
        }
        
        return df_resampled, metadata
    
    def apply_random_oversampling(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Random Oversampling (duplicate minority class samples).
        
        Returns:
            Tuple of (sampled_df, metadata)
        """
        # Get class distribution
        class_counts = df[target_column].value_counts()
        max_count = class_counts.max()
        
        # Oversample each class to match the majority
        dfs_resampled = []
        for class_val in class_counts.index:
            class_df = df[df[target_column] == class_val]
            current_count = len(class_df)
            
            if current_count < max_count:
                # Oversample
                n_samples_needed = max_count - current_count
                oversampled = class_df.sample(
                    n=n_samples_needed,
                    replace=True,
                    random_state=42
                )
                dfs_resampled.append(pd.concat([class_df, oversampled], ignore_index=True))
            else:
                dfs_resampled.append(class_df)
        
        # Combine
        df_resampled = pd.concat(dfs_resampled, ignore_index=True)
        df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Get before/after stats
        before_dist = self.analyze_class_distribution(df, target_column)
        after_dist = self.analyze_class_distribution(df_resampled, target_column)
        
        metadata = {
            'method': 'random_oversampling',
            'before_distribution': before_dist,
            'after_distribution': after_dist,
            'samples_added': len(df_resampled) - len(df)
        }
        
        return df_resampled, metadata
    
    def apply_random_undersampling(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Random Undersampling (remove majority class samples).
        
        Returns:
            Tuple of (sampled_df, metadata)
        """
        # Get class distribution
        class_counts = df[target_column].value_counts()
        min_count = class_counts.min()
        
        # Undersample each class to match the minority
        dfs_resampled = []
        for class_val in class_counts.index:
            class_df = df[df[target_column] == class_val]
            current_count = len(class_df)
            
            if current_count > min_count:
                # Undersample
                class_df = class_df.sample(
                    n=min_count,
                    random_state=42
                )
            
            dfs_resampled.append(class_df)
        
        # Combine
        df_resampled = pd.concat(dfs_resampled, ignore_index=True)
        df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Get before/after stats
        before_dist = self.analyze_class_distribution(df, target_column)
        after_dist = self.analyze_class_distribution(df_resampled, target_column)
        
        metadata = {
            'method': 'random_undersampling',
            'before_distribution': before_dist,
            'after_distribution': after_dist,
            'samples_removed': len(df) - len(df_resampled)
        }
        
        return df_resampled, metadata
