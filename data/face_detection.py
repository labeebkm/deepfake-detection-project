"""
Face detection and alignment utilities.
"""

import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class FaceDetector:
    """Face detection and alignment utilities."""
    
    def __init__(self, method: str = 'mtcnn', min_face_size: int = 40, margin: float = 0.25):
        """
        Initialize FaceDetector.
        
        Args:
            method: Detection method ('mtcnn' or 'tensorflow')
            min_face_size: Minimum face size
            margin: Margin around detected face
        """
        self.method = method
        self.min_face_size = min_face_size
        self.margin = margin
        
        if method == 'mtcnn' and MTCNN_AVAILABLE:
            self.detector = MTCNN(min_face_size=min_face_size)
        else:
            self.detector = None
            # Use TensorFlow's face detection or OpenCV
            self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face detection results with bounding boxes
        """
        if self.method == 'mtcnn' and self.detector is not None:
            return self._detect_mtcnn(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN."""
        if len(image.shape) == 3:
            # Convert RGB to BGR for MTCNN
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        detections = self.detector.detect_faces(image_bgr)
        
        results = []
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            keypoints = detection.get('keypoints', {})
            
            results.append({
                'box': [box[0], box[1], box[0] + box[2], box[1] + box[3]],  # [x1, y1, x2, y2]
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        return results
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'box': [x, y, x + w, y + h],
                'confidence': 1.0,  # Haar cascade doesn't provide confidence
                'keypoints': {}
            })
        
        return results
    
    def extract_face(self, image: np.ndarray, bbox: List[int], 
                    target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract and align face from image.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            target_size: Target size for extracted face (height, width)
            
        Returns:
            Extracted face image
        """
        x1, y1, x2, y2 = bbox
        
        # Add margin
        h, w = image.shape[:2]
        margin_x = int((x2 - x1) * self.margin)
        margin_y = int((y2 - y1) * self.margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        # Resize if target size specified
        if target_size:
            face = cv2.resize(face, target_size)
        
        return face
    
    def align_face(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Align face using keypoints.
        
        Args:
            image: Input image
            keypoints: Facial keypoints dictionary
            
        Returns:
            Aligned face image
        """
        if not keypoints or 'left_eye' not in keypoints or 'right_eye' not in keypoints:
            return image
        
        # Get eye positions
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        return aligned







