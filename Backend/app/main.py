from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import upload, eda, preprocessing, training, export

app = FastAPI(title="ClickTrain ML Backend", description="Machine Learning Platform Backend")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(eda.router, prefix="/api", tags=["eda"])
app.include_router(preprocessing.router, prefix="/api", tags=["preprocessing"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(export.router, prefix="/api", tags=["export"])

@app.get("/")
async def root():
    return {"message": "ClickTrain ML Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}