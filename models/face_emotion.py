import cv2
from deepface import DeepFace
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FastEmotionAnalyzer:
    """Face emotion analyzer optimized for orchestrator integration"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.current_emotion = "unknown"
        self.current_confidence = 0.0
        self.initialized = False
        
        # Initialize and test
        self._initialize()
    
    def _initialize(self):
        """Initialize and test emotion detection"""
        try:
            # Test with small image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            DeepFace.analyze(
                test_image,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            self.initialized = True
            logger.info("Face emotion analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Face emotion analyzer initialization failed: {e}")
            self.initialized = False
    
    def detect_best_face(self, frame):
        """Detect the best face for emotion analysis"""
        if not self.initialized:
            return None
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60)
            )
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            best_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = best_face
            
            # Expand region slightly
            expansion = int(0.1 * max(w, h))
            x = max(0, x - expansion)
            y = max(0, y - expansion)
            w = min(frame.shape[1] - x, w + 2 * expansion)
            h = min(frame.shape[0] - y, h + 2 * expansion)
            
            return (x, y, w, h)
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None
    
    def enhance_face_for_emotion(self, face_image):
        """Enhanced preprocessing for emotion detection"""
        try:
            height, width = face_image.shape[:2]
            
            if width < 96 or height < 96:
                face_image = cv2.resize(face_image, (96, 96), interpolation=cv2.INTER_CUBIC)
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            enhanced_face = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
            normalized = cv2.convertScaleAbs(enhanced_face, alpha=1.2, beta=10)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Face enhancement failed, using original: {e}")
            return face_image
    
    def analyze_emotion(self, video_frame) -> Optional[Dict[str, Any]]:
        """Main analysis function for orchestrator integration"""
        if not self.initialized:
            logger.warning("Face emotion analyzer not initialized")
            return None
        
        try:
            # Detect face in frame
            face_coords = self.detect_best_face(video_frame)
            if not face_coords:
                return None
            
            # Extract face region
            x, y, w, h = face_coords
            face_roi = video_frame[y:y + h, x:x + w]
            
            # Enhance face
            enhanced_face = self.enhance_face_for_emotion(face_roi)
            
            # Analyze emotion
            result = DeepFace.analyze(
                enhanced_face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Extract emotion data
            if isinstance(result, list) and len(result) > 0:
                emotion_data = result[0]['emotion']
                dominant = result[0]['dominant_emotion']
            else:
                emotion_data = result['emotion']
                dominant = result['dominant_emotion']
            
            # Update current state - DeepFace returns percentages (0-100), convert to 0-1 for frontend
            self.current_emotion = dominant
            self.current_confidence = emotion_data.get(dominant, 0.0) / 100.0  # Convert percentage to ratio
            
            # Convert all emotion scores from percentages to ratios
            emotion_scores_normalized = {k: v / 100.0 for k, v in emotion_data.items()}
            
            return {
                'emotion': dominant,
                'confidence': self.current_confidence,  # Now 0-1 range
                'emotion_scores': emotion_scores_normalized,  # Now 0-1 range
                'face_coordinates': face_coords,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Face emotion analysis failed: {e}")
            return None
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current emotion state"""
        return {
            'emotion': self.current_emotion,
            'confidence': self.current_confidence,
            'initialized': self.initialized
        }