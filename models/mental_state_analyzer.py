import logging
import torch
import traceback
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MentalStateAnalyzer:
    """Clinical mental state analysis optimized for orchestrator integration"""
    
    def __init__(self, model_path="textmodels/mental"):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
    
    def load_model(self) -> bool:
        """Load mental state analysis model"""
        logger.info("Loading clinical mental state model...")
        
        try:
            if not self.model_path.exists():
                logger.error(f"Mental state model path does not exist: {self.model_path}")
                return False
            
            config_file = self.model_path / "config.json"
            if not config_file.exists():
                logger.error(f"Mental state model config missing: {config_file}")
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
            
            logger.info(f"Mental state model loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load mental state model: {str(e)}")
            logger.debug(traceback.format_exc())
            self.loaded = False
            return False
    
    def analyze_mental_state(self, text: str) -> Dict[str, Any]:
        """Analyze mental state from text"""
        if not self.loaded or not self.model or not text or len(text.strip()) < 3:
            return {
                'mental_state': 'Insufficient Text',
                'confidence': 0.0,
                'details': 'Not enough text for analysis or model not loaded'
            }
        
        try:
            # Process text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = outputs.logits.argmax().item()
                confidence = predictions.max().item()
            
            # Clinical interpretation
            if confidence < 0.5:
                mental_state = "Unclear/Neutral"
            elif predicted_class_id == 0:
                mental_state = "Normal/Stable"
            elif predicted_class_id == 1:
                mental_state = "Mild Concern"
            else:
                mental_state = "Requires Attention"
            
            # Additional indicators
            indicators = self._analyze_text_indicators(text)
            
            return {
                'mental_state': mental_state,
                'confidence': confidence,
                'indicators': indicators,
                'text_analyzed': text,
                'raw_prediction': predicted_class_id,
                'model_device': self.device
            }
            
        except Exception as e:
            logger.error(f"Mental state analysis failed: {str(e)}")
            return {
                'mental_state': 'Analysis Error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_text_indicators(self, text: str) -> List[str]:
        """Analyze text for specific mental health indicators"""
        text_lower = text.lower()
        indicators = []
        
        depression_keywords = ['sad', 'depressed', 'hopeless', 'empty', 'tired', 'worthless', 'down']
        anxiety_keywords = ['worried', 'anxious', 'nervous', 'panic', 'fear', 'stressed', 'overwhelmed']
        
        if any(keyword in text_lower for keyword in depression_keywords):
            indicators.append('Depression signs')
        
        if any(keyword in text_lower for keyword in anxiety_keywords):
            indicators.append('Anxiety signs')
        
        return indicators
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded