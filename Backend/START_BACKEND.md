# Starting the Backend Server

## Quick Start

To start the backend server on port 8001:

```bash
cd Backend
python run_server.py
```

Or using uvicorn directly:

```bash
cd Backend
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

## Verify Backend is Running

1. **Check Health Endpoint:**
   Open your browser and go to: `http://127.0.0.1:8001/health`
   
   You should see: `{"status":"healthy"}`

2. **Check API Documentation:**
   Open: `http://127.0.0.1:8001/docs`
   
   This shows the interactive API documentation (Swagger UI)

3. **Check Root Endpoint:**
   Open: `http://127.0.0.1:8001/`
   
   You should see: `{"message":"ClickTrain ML Backend is running"}`

## Troubleshooting

### Port Already in Use
If port 8001 is already in use:
```bash
# Windows
netstat -ano | findstr :8001
# Then kill the process using the PID

# Linux/Mac
lsof -i :8001
kill -9 <PID>
```

### Dependencies Not Installed
```bash
cd Backend
pip install -r requirements.txt
```

### CORS Issues
The backend is configured to allow all origins. If you have CORS issues, check:
- Backend is running on `127.0.0.1:8001`
- Frontend is accessing from the same origin or the backend CORS settings

## Testing the Preprocessing Endpoints

Once the backend is running, you can test the preprocessing endpoints:

1. **Upload a dataset first** (via frontend or API)
2. **Get dataset stats:**
   ```
   GET http://127.0.0.1:8001/api/preprocessing/{session_id}/stats
   ```
3. **Analyze missing values:**
   ```
   GET http://127.0.0.1:8001/api/preprocessing/{session_id}/missing-values
   ```

## Production Deployment

For production, use:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

Or use a process manager like `gunicorn` with uvicorn workers.
