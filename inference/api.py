"""
FastAPI REST API for deepfake detection.
"""

import base64
import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from utils.visualization import generate_gradcam, overlay_heatmap
from .detector import DeepfakeDetector


def _encode_png_base64(rgb_image: np.ndarray) -> str:
    """Encode an RGB numpy image to a data URI base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(rgb_image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def create_app(
    weights_path: str,
    config_path: str,
    confidence_threshold: float = 0.5,
    cors_origins: Optional[List[str]] = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        weights_path: Path to saved weights (e.g. best_model.weights.h5)
        config_path: Path to config.yaml (used to rebuild the model)
        confidence_threshold: Confidence threshold
        cors_origins: Allowed CORS origins; defaults to ["*"]

    Returns:
        FastAPI application instance
    """
    app = FastAPI(title="Deepfake Detection API")
    ui_dir = Path(__file__).resolve().parent / "static"
    ui_path = ui_dir / "index.html"

    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    async def predict(
        file: UploadFile = File(...),
        explain: bool = Query(False, description="Include Grad-CAM heatmap in response"),
    ):
        """
        Predict if uploaded image is deepfake.

        Args:
            file: Uploaded image file
            explain: Whether to include Grad-CAM overlay

        Returns:
            Prediction results (+ optional Grad-CAM heatmap)
        """
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload a valid image file.")

        # Read image
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_np = np.array(image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

        # Predict
        try:
            result = detector.predict(image_np)

            # Generate Grad-CAM only when requested to avoid extra compute overhead.
            if explain:
                try:
                    image_batch, feature_batch = detector._preprocess_image(image_np)
                    cam = generate_gradcam(
                        model_backbone=detector.model.rgb_backbone,
                        preprocessed_image=image_batch,
                        layer_name="top_conv",
                        classifier_model=detector.model,
                        feature_vector=feature_batch,
                        class_idx=1,
                    )
                    overlay = overlay_heatmap(image_np, cam)
                    result["gradcam_heatmap"] = _encode_png_base64(overlay)
                except Exception as gradcam_exc:
                    # Keep prediction response successful even if Grad-CAM fails.
                    result["gradcam_error"] = str(gradcam_exc)

            return JSONResponse(content=result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
