"""
FastAPI REST API for deepfake detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io
from pathlib import Path

from .detector import DeepfakeDetector


def create_app(
    weights_path: str,
    config_path: str,
    confidence_threshold: float = 0.5,
) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        weights_path: Path to saved weights (e.g. best_model.weights.h5)
        config_path: Path to config.yaml (used to rebuild the model)
        confidence_threshold: Confidence threshold
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(title="Deepfake Detection API")
    ui_dir = Path(__file__).resolve().parent / "static"
    ui_path = ui_dir / "index.html"

    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")
    
    # Initialize detector
    detector = DeepfakeDetector(
        weights_path=weights_path,
        config_path=config_path,
        confidence_threshold=confidence_threshold,
    )
    
    @app.get("/", response_class=FileResponse)
    def root():
        """Web UI."""
        if not ui_path.exists():
            return HTMLResponse(
                "<h2>UI file not found.</h2><p>Expected: inference/static/index.html</p>",
                status_code=404,
            )
        return FileResponse(str(ui_path))

    @app.get("/api")
    def api_root():
        """API status endpoint."""
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
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload a valid image file.")

        # Read image
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image = np.array(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
        
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

