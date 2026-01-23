# ClickTrain ML Backend

A backend-only Machine Learning platform built with FastAPI that handles the complete ML workflow: data upload, EDA, preprocessing, model training, and export.

## Why Docker?

Docker is used to containerize the ML application for several important reasons:

### 1. **Environment Consistency**
- Ensures the same Python version (3.10) and dependencies run everywhere
- No "works on my machine" issues
- All ML libraries (pandas, scikit-learn, matplotlib) are pre-installed

### 2. **Isolation**
- ML computations run in their own container
- Doesn't interfere with other applications on the server
- Memory and CPU usage is contained

### 3. **Easy Deployment**
- Single command to run the entire ML platform
- Can be deployed to cloud platforms (AWS, GCP, Azure)
- Scales horizontally for multiple users

### 4. **Dependency Management**
- All Python packages are installed in the container
- No need to manage virtual environments manually
- Faster startup than installing dependencies each time

## Backend Flow

The ML workflow follows these steps:

### 1. Dataset Upload (`POST /api/upload`)
- User uploads CSV file
- Backend validates file type (.csv only)
- Stores data in memory with unique session ID
- Returns session ID for subsequent operations

### 2. Exploratory Data Analysis (`GET /api/eda/{session_id}`)
- Analyzes dataset structure (shape, columns, data types)
- Calculates missing values and basic statistics
- Generates plots: histograms and correlation heatmap
- Returns JSON statistics + image file paths

### 3. Data Preprocessing (`POST /api/preprocessing/{session_id}`)
- Handles missing values (mean/median/mode/drop strategies)
- Encodes categorical features with Label Encoding
- Applies feature scaling (StandardScaler) if requested
- Stores processed data for training

### 4. Model Training (`POST /api/training/{session_id}`)
- Supports: Linear Regression, Logistic Regression, Random Forest
- Automatically detects classification vs regression
- Calculates performance metrics (accuracy, precision, etc.)
- Stores trained model in memory

### 5. Model Export (`GET /api/export/{session_id}`)
- Saves trained model as .pkl file
- Returns downloadable file to user

## Multi-User Support

- Each user gets a unique session ID
- All data stored in RAM (no database needed)
- Sessions are isolated - users can't access others' data
- Temporary files (plots, models) cleaned up automatically

## Running the Application

### With Docker (Recommended)

1. Build the container:
```bash
docker build -t clicktrain-backend .
```

2. Run the container:
```bash
docker run -p 8000:8000 clicktrain-backend
```

3. Access API at: http://localhost:8000

### Without Docker (Development)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

## Tech Stack

- **Python 3.10**: Modern Python with great ML support
- **FastAPI**: High-performance async web framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Plot generation
- **Scikit-learn**: Machine learning algorithms
- **Docker**: Containerization for deployment

## Project Structure

```
Backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routes/              # API endpoints
│   │   ├── upload.py
│   │   ├── eda.py
│   │   ├── preprocessing.py
│   │   ├── training.py
│   │   └── export.py
│   ├── services/            # Business logic
│   │   ├── data_service.py
│   │   ├── eda_service.py
│   │   ├── preprocessing_service.py
│   │   ├── training_service.py
│   │   └── export_service.py
│   ├── models/              # Data models (future use)
│   └── utils/               # Utility functions (future use)
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
└── README.md               # This file
```