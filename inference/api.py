"""
FastAPI REST API for deepfake detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
from typing import Optional

from .detector import DeepfakeDetector


def create_app(model_path: str, confidence_threshold: float = 0.5) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_path: Path to saved model
        confidence_threshold: Confidence threshold
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(title="Deepfake Detection API")
    
    # Initialize detector
    detector = DeepfakeDetector(model_path, confidence_threshold)
    
    @app.get("/")
    def root():
        """Root endpoint."""
        return {"message": "Deepfake Detection API", "status": "running"}
    
    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        """
        Predict if uploaded image is deepfake.
        
        Args:
            file: Uploaded image file
            
        Returns:
            Prediction results
        """
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = np.array(image)
        
        # Convert RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Predict
        try:
            result = detector.predict(image)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app








