"""
Serve model via FastAPI.
"""

import argparse
import uvicorn
from inference.api import create_app


def main():
    parser = argparse.ArgumentParser(description='Serve deepfake detection model via API')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(args.model_path, args.confidence_threshold)
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()








