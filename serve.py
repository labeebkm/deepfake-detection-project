"""
Serve model via FastAPI.
"""

import argparse
import uvicorn
from inference.api import create_app


def main():
    parser = argparse.ArgumentParser(description="Serve deepfake detection model via API")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved weights (e.g. ./checkpoints/best_model.weights.h5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file (used to infer feature_dim for the three-stream model)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--cors_origins",
        type=str,
        default="*",
        help="Comma-separated CORS origins (e.g. http://localhost:5500,http://127.0.0.1:5500)",
    )

    args = parser.parse_args()

    cors_origins = [origin.strip() for origin in args.cors_origins.split(",") if origin.strip()]

    app = create_app(
        weights_path=args.model_path,
        config_path=args.config,
        confidence_threshold=args.confidence_threshold,
        cors_origins=cors_origins,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
