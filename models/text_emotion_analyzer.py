import json 
import logging
import torch
import traceback
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TextEmotionAnalyzer:
    """Text-based emotion analysis optimized for orchestrator integration"""
    
    def __init__(self, model_path="voicemodels/text_emotion"):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        config_path = self.model_path / "model_info.json"
        default_config = {
            'emotions': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'],
            'clinical_mapping': {
                'anger': 'Anger/Irritation',
                'disgust': 'Disgust/Aversion',
                'fear': 'Fear/Anxiety',
                'joy': 'Joy/Happiness',
                'neutral': 'Neutral/Calm',
                'sadness': 'Sadness/Melancholy',
                'surprise': 'Surprise/Shock'
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def load_model(self) -> bool:
        """Load text emotion model"""
        logger.info("Loading text emotion model...")
        
        try:
            if not self.model_path.exists():
                logger.error(f"Text model path does not exist: {self.model_path}")
                return False
            
            config_file = self.model_path / "config.json"
            if not config_file.exists():
                logger.error(f"Text model config missing: {config_file}")
                return False
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), 
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path), 
                local_files_only=True
            )
            
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            
            logger.info(f"Text emotion model loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Text emotion model failed: {str(e)}")
            logger.debug(traceback.format_exc())
            self.loaded = False
            return False
    
    def analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze emotion from text content"""
        if not self.loaded or not self.model or not text or len(text.strip()) < 3:
            return {
                'emotion': 'Neutral/Calm',
                'confidence': 0.0,
                'source': 'text',
                'error': 'Text model not available or insufficient text'
            }
        
        try:
            # Process text with speed optimization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256  # Reduced for faster processing
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = outputs.logits.argmax().item()
                confidence = predictions.max().item()
            
            # Get emotion labels
            text_emotions = self.config.get('emotions',
                ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
            
            raw_emotion = text_emotions[predicted_class_id] if predicted_class_id < len(text_emotions) else 'neutral'
            
            # Map to clinical terms
            clinical_mapping = self.config.get('clinical_mapping', {})
            clinical_emotion = clinical_mapping.get(raw_emotion, raw_emotion.title())
            
            return {
                'emotion': clinical_emotion,
                'confidence': confidence,
                'raw_emotion': raw_emotion,
                'source': 'text',
                'text_analyzed': text,
                'all_scores': predictions[0].cpu().numpy().tolist(),
                'model_device': self.device
            }
            
        except Exception as e:
            logger.error(f"Text emotion analysis failed: {str(e)}")
            return {
                'emotion': 'Neutral/Calm',
                'confidence': 0.0,
                'source': 'text',
                'error': str(e)
            }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded