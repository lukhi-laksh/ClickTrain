# Data Preprocessing Backend Architecture

## Overview

This document describes the complete backend architecture for the Data Preprocessing module of ClickTrain. The system is designed to be production-ready, modular, scalable, and maintainable.

## Architecture Principles

1. **Modularity**: Each preprocessing operation is handled by a dedicated service
2. **Version Control**: Every action creates a new dataset version with undo/redo support
3. **Audit Trail**: All actions are logged for reproducibility
4. **Separation of Concerns**: Clear boundaries between data management, processing, and API layers
5. **Error Handling**: Graceful error handling with clear error messages
6. **Persistence**: Encoders and scalers are stored for reuse during training

## Core Components

### 1. DatasetManager (`dataset_manager.py`)

**Responsibility**: Manages dataset versions, original dataset preservation, and metadata tracking.

**Key Features**:
- Maintains original dataset (immutable)
- Maintains current dataset (mutable)
- Version history stack
- Undo/Redo functionality
- Metadata tracking (shape, columns, dtypes)

**Key Methods**:
- `initialize_session()`: Initialize a new dataset session
- `get_original()`: Get the original immutable dataset
- `get_current()`: Get the current mutable dataset
- `create_version()`: Create a new dataset version
- `undo()`: Undo the last action
- `redo()`: Redo the last undone action
- `reset_to_original()`: Reset to original state

### 2. NullValueHandler (`null_value_handler.py`)

**Responsibility**: Comprehensive null value detection and handling.

**Key Features**:
- Detects multiple null representations: NaN, NULL, None, "", "NaN", "nan", "N/A", etc.
- Per-column null statistics
- Multiple imputation strategies:
  - Drop rows
  - Mean/Median (numerical)
  - Mode (categorical)
  - Constant value

**Key Methods**:
- `detect_null_values()`: Analyze null values in dataset
- `handle_missing_values()`: Apply imputation strategy

### 3. DuplicateHandler (`duplicate_handler.py`)

**Responsibility**: Detects and handles duplicate rows and columns.

**Key Features**:
- Detects duplicate rows
- Detects duplicate columns
- Preview of duplicate rows
- Options: keep first, keep last

**Key Methods**:
- `detect_duplicates()`: Analyze duplicates
- `remove_duplicates()`: Remove duplicates with options

### 4. ConstantColumnDetector (`constant_column_detector.py`)

**Responsibility**: Detects constant and low-variance columns.

**Key Features**:
- Detects zero-variance columns
- Shows constant value per column
- Selective column removal

**Key Methods**:
- `detect_constant_columns()`: Find constant columns
- `remove_columns()`: Remove specified columns

### 5. EncoderManager (`encoder_manager.py`)

**Responsibility**: Manages all categorical encoding operations.

**Supported Encodings**:
- **Label Encoding**: Integer labels (0, 1, 2...)
- **One-Hot Encoding**: Binary columns per category
  - Options: drop_first, handle_binary separately
- **Ordinal Encoding**: Manual or auto category ordering
- **Target Encoding**: Mean encoding (with leakage warning)

**Key Features**:
- Per-column encoding selection
- Encoder persistence for training
- Serialization support

**Key Methods**:
- `label_encode()`: Apply Label Encoding
- `one_hot_encode()`: Apply One-Hot Encoding
- `ordinal_encode()`: Apply Ordinal Encoding
- `target_encode()`: Apply Target Encoding
- `serialize_encoders()`: Serialize for persistence

### 6. ScalerManager (`scaler_manager.py`)

**Responsibility**: Manages feature scaling operations.

**Supported Scalers**:
- **StandardScaler**: Z-score normalization (mean=0, std=1)
- **MinMaxScaler**: Min-max normalization (range [0, 1])
- **RobustScaler**: Median and IQR based (outlier-resistant)

**Key Features**:
- Column-specific scaling
- Before/after statistics
- Scaler persistence for training
- Serialization support

**Key Methods**:
- `scale_features()`: Apply scaling to columns
- `serialize_scalers()`: Serialize for persistence

### 7. OutlierHandler (`outlier_handler.py`)

**Responsibility**: Detects and handles outliers.

**Detection Methods**:
- **IQR Method**: Q1 - 1.5×IQR to Q3 + 1.5×IQR
- **Z-Score Method**: |z| > threshold (default 3.0)

**Actions**:
- Remove outlier rows
- Cap values (Winsorization)
- Add outlier flag column

**Key Methods**:
- `detect_outliers()`: Detect outliers
- `handle_outliers()`: Apply outlier treatment

### 8. SamplingHandler (`sampling_handler.py`)

**Responsibility**: Handles class imbalance through sampling.

**Supported Methods**:
- **SMOTE**: Synthetic Minority Oversampling
- **Random Oversampling**: Duplicate minority samples
- **Random Undersampling**: Remove majority samples

**Important**: Sampling should only be applied to training data after train-test split.

**Key Methods**:
- `analyze_class_distribution()`: Analyze class balance
- `apply_smote()`: Apply SMOTE
- `apply_random_oversampling()`: Random oversampling
- `apply_random_undersampling()`: Random undersampling

### 9. AuditLogger (`audit_logger.py`)

**Responsibility**: Tracks all preprocessing actions for audit and reproducibility.

**Key Features**:
- Action logging with timestamps
- Success/failure tracking
- Metadata storage
- Action summaries

**Key Methods**:
- `log_action()`: Log an action
- `get_logs()`: Retrieve logs
- `get_action_summary()`: Get summary statistics

### 10. PreprocessingEngine (`preprocessing_engine.py`)

**Responsibility**: Main orchestrator that coordinates all preprocessing services.

**Key Features**:
- Coordinates all preprocessing operations
- Manages dataset versions
- Integrates audit logging
- Provides unified API

**Key Methods**:
- All preprocessing operations (missing values, duplicates, encoding, etc.)
- Version control (undo, redo, reset)
- Summary and export functions

## API Design

### Endpoint Structure

All preprocessing endpoints follow the pattern:
```
/api/preprocessing/{session_id}/{operation}
```

### Request/Response Patterns

**Analysis Endpoints** (GET):
- Return statistics and analysis results
- No dataset modification

**Action Endpoints** (POST):
- Modify the dataset
- Create new version
- Return version info and metadata

### Key Endpoints

#### Missing Values
- `GET /preprocessing/{session_id}/missing-values`: Analyze missing values
- `POST /preprocessing/{session_id}/missing-values`: Handle missing values

#### Duplicates
- `GET /preprocessing/{session_id}/duplicates`: Analyze duplicates
- `POST /preprocessing/{session_id}/duplicates`: Remove duplicates

#### Constant Columns
- `GET /preprocessing/{session_id}/constant-columns`: Detect constant columns
- `POST /preprocessing/{session_id}/constant-columns`: Remove constant columns

#### Encoding
- `POST /preprocessing/{session_id}/encoding/label`: Label encoding
- `POST /preprocessing/{session_id}/encoding/onehot`: One-hot encoding
- `POST /preprocessing/{session_id}/encoding/ordinal`: Ordinal encoding
- `POST /preprocessing/{session_id}/encoding/target`: Target encoding

#### Scaling
- `POST /preprocessing/{session_id}/scaling`: Apply feature scaling

#### Outliers
- `POST /preprocessing/{session_id}/outliers/detect`: Detect outliers
- `POST /preprocessing/{session_id}/outliers/handle`: Handle outliers

#### Sampling
- `GET /preprocessing/{session_id}/sampling/distribution`: Class distribution
- `POST /preprocessing/{session_id}/sampling`: Apply sampling

#### Version Control
- `GET /preprocessing/{session_id}/stats`: Dataset statistics
- `GET /preprocessing/{session_id}/history`: Action history
- `POST /preprocessing/{session_id}/undo`: Undo last action
- `POST /preprocessing/{session_id}/redo`: Redo last undone action
- `POST /preprocessing/{session_id}/reset`: Reset to original

#### Summary & Export
- `GET /preprocessing/{session_id}/summary`: Comprehensive summary
- `GET /preprocessing/{session_id}/encoders`: Get encoders
- `GET /preprocessing/{session_id}/scalers`: Get scalers

## Data Versioning Strategy

### Version Stack Structure

```
Original Dataset (Version 0) [Immutable]
    ↓
Version 1: Missing values handled
    ↓
Version 2: Duplicates removed
    ↓
Version 3: Encoding applied
    ↓
Current Version (Version N)
```

### Undo/Redo Mechanism

- **Undo Stack**: Contains version IDs in order of application
- **Redo Stack**: Contains undone version IDs
- When a new action is performed, redo stack is cleared
- Each undo/redo operation restores the exact dataset state

### Version Metadata

Each version stores:
- Dataset snapshot (DataFrame copy)
- Version ID
- Action type
- Action metadata
- Timestamp
- Shape and column information

## Storage Strategy

### In-Memory Storage

- **Original Datasets**: Immutable copies
- **Current Datasets**: Mutable working copies
- **Version History**: List of DatasetVersion objects
- **Encoders/Scalers**: Dictionary of sklearn objects

### Persistence for Training

Encoders and scalers are serialized using:
- **Pickle**: For sklearn objects
- **Base64**: For JSON-safe transmission
- **Metadata**: Stored as dictionaries

### Session Management

- Each session is isolated
- Session data is cleared when session ends
- Session ID is generated on dataset upload

## Error Handling

### Error Types

1. **ValueError**: Invalid input (e.g., column not found)
2. **HTTPException**: API-level errors with status codes
3. **Exception**: Unexpected errors with detailed messages

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Scenarios

- Session not found: 404
- Invalid column name: 404
- Invalid operation: 400
- Processing failure: 500

## Best Practices

### For ML Correctness

1. **Train-Test Split First**: Always split before sampling
2. **Target Encoding**: Use with cross-validation to prevent leakage
3. **Outlier Removal**: Consider domain knowledge before removing
4. **Constant Columns**: Remove before training
5. **Missing Values**: Choose strategy based on data characteristics

### For Reproducibility

1. **Version Control**: Every action creates a version
2. **Audit Logging**: All actions are logged
3. **Encoder/Scaler Persistence**: Store for inference
4. **Metadata Tracking**: Track all transformations

### For Performance

1. **Copy Operations**: Use `.copy()` to avoid modifying originals
2. **Memory Management**: Clear sessions when done
3. **Efficient Operations**: Use vectorized pandas operations

## Integration with Training

### Encoder/Scaler Reuse

Encoders and scalers are stored per session and can be retrieved for:
- Training data transformation
- Test data transformation
- Inference on new data

### Preprocessing Summary

The summary endpoint provides:
- Complete action history
- Serialized encoders/scalers
- Dataset statistics
- Ready for training pipeline

## Future Enhancements

1. **Persistence Layer**: Database storage for versions
2. **Distributed Processing**: Support for large datasets
3. **Custom Transformers**: User-defined preprocessing steps
4. **Pipeline Export**: Export preprocessing pipeline as code
5. **Validation**: Data validation before/after preprocessing

## Testing Considerations

1. **Unit Tests**: Each service should have unit tests
2. **Integration Tests**: Test full preprocessing workflows
3. **Version Control Tests**: Test undo/redo functionality
4. **Error Handling Tests**: Test all error scenarios
5. **Performance Tests**: Test with large datasets

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Preprocessing transformers
- `imbalanced-learn`: SMOTE and sampling
- `fastapi`: API framework
- `pydantic`: Request validation
