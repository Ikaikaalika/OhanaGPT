from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .config import config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    description="AI-powered genealogy parent prediction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {config.APP_NAME}",
        "version": "1.0.0"
    }

@app.on_event("startup")
async def startup():
    """Startup tasks."""
    # Create necessary directories
    config.UPLOAD_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)
    
    logging.info(f"{config.APP_NAME} started successfully")

@app.on_event("shutdown")
async def shutdown():
    """Shutdown tasks."""
    logging.info(f"{config.APP_NAME} shutting down")