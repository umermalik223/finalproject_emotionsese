import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from transformers import pipeline
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class OptimizedSERAnalyzer:
    """Speech emotion analyzer optimized for orchestrator integration"""
    
    def __init__(self, model_path="voicemodels/speech_emotion"):
        self.sample_rate = 16000
        self.emotion_pipeline = None
        self.model_name = None
        self.model_path = Path(model_path) if model_path else None
        self.initialized = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize emotion model - try local first, then remote"""
        try:
            logger.info("Loading speech emotion model...")
            
            # First try to use local model if it exists
            if self.model_path and self.model_path.exists():
                try:
                    logger.info(f"Attempting to load local speech emotion model: {self.model_path}")
                    self.emotion_pipeline = pipeline(
                        "audio-classification",
                        model=str(self.model_path.absolute()),
                        device=-1,  # CPU for stability
                        return_all_scores=True
                    )
                    self.model_name = f"local_{self.model_path.name}"
                    logger.info(f"✅ Local speech emotion model loaded: {self.model_path.name}")
                    self.initialized = True
                    return
                except Exception as local_error:
                    logger.warning(f"Local model failed, will try remote: {local_error}")
            
            # Fallback to remote HuggingFace model (cached after first download)
            logger.info("Using cached HuggingFace model: superb/wav2vec2-base-superb-er")
            model_name = "superb/wav2vec2-base-superb-er"
            
            self.emotion_pipeline = pipeline(
                "audio-classification",
                model=model_name,
                device=-1,  # CPU for stability  
                return_all_scores=True
            )
            
            self.model_name = model_name
            self.initialized = True
            logger.info(f"✅ Speech emotion model loaded (cached): {model_name.split('/')[-1]}")
            
        except Exception as e:
            logger.error(f"Failed to load speech emotion model: {e}")
            self.initialized = False
    
    def preprocess_audio(self, audio_data):
        """Preprocess audio for analysis"""
        try:
            # Quick silence check
            if np.max(np.abs(audio_data)) < 0.008:
                return None
            
            # Normalize
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Simple noise gate
            audio_data = np.where(np.abs(audio_data) < 0.01, 0, audio_data)
            
            # Ensure consistent length (3 seconds)
            target_length = int(self.sample_rate * 3.0)
            if len(audio_data) != target_length:
                if len(audio_data) < target_length:
                    audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
                else:
                    start = (len(audio_data) - target_length) // 2
                    audio_data = audio_data[start:start + target_length]
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return None
    
    def analyze_audio_chunk(self, audio_chunk) -> Optional[Dict[str, Any]]:
        """Main analysis function for orchestrator integration"""
        if not self.initialized or self.emotion_pipeline is None:
            logger.warning("Speech emotion analyzer not initialized")
            return None
        
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_chunk)
            if processed_audio is None:
                return None
            
            # Run emotion analysis
            results = self.emotion_pipeline(processed_audio, sampling_rate=self.sample_rate)
            
            # Process results
            emotion_scores = {
                'angry': 0.0, 'disgust': 0.0, 'fear': 0.0,
                'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0
            }
            
            # Map model outputs to standard emotions
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'ang' in label:
                    emotion_scores['angry'] += score
                elif 'hap' in label or 'joy' in label:
                    emotion_scores['happy'] += score
                elif 'sad' in label:
                    emotion_scores['sad'] += score
                elif 'fear' in label:
                    emotion_scores['fear'] += score
                elif 'disgust' in label:
                    emotion_scores['disgust'] += score
                elif 'surprise' in label:
                    emotion_scores['surprise'] += score
                else:
                    emotion_scores['neutral'] += score
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]  # Keep as 0-1 range for frontend
            
            # Only return if confidence is reasonable (0.4 = 40%)
            if confidence < 0.4:
                return None
            
            return {
                'emotion': dominant_emotion,
                'emotion_scores': emotion_scores,
                'confidence': confidence,
                'model_used': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Speech emotion analysis failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'initialized': self.initialized,
            'sample_rate': self.sample_rate
        }