import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EmotionFusionEngine:
    """Advanced multimodal emotion fusion with weighted confidence scoring"""
    
    def __init__(self):
        # Modality weights based on reliability for therapeutic applications
        self.modality_weights = {
            'face': 0.35,     # Visual cues are strong indicators
            'speech': 0.30,   # Vocal patterns very reliable
            'text': 0.25,     # Text content analysis
            'mental_state': 0.10  # Clinical assessment
        }
        
        # Emotion mapping for consistency across modalities
        self.emotion_mapping = {
            'happy': ['happy', 'joy', 'positive', 'excited'],
            'sad': ['sad', 'sadness', 'depressed', 'down'],
            'angry': ['angry', 'anger', 'frustrated', 'irritated'],
            'fear': ['fear', 'anxiety', 'worried', 'nervous'],
            'neutral': ['neutral', 'calm', 'normal'],
            'surprise': ['surprise', 'surprised', 'shocked'],
            'disgust': ['disgust', 'disgusted', 'repulsed']
        }
    
    def fuse_emotions(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse multimodal emotion data into unified assessment"""
        try:
            # Extract individual modality results
            face_data = multimodal_data.get('face_emotion', {})
            speech_data = multimodal_data.get('speech_emotion', {})
            text_data = multimodal_data.get('text_emotion', {})
            mental_data = multimodal_data.get('mental_state', {})
            
            # Normalize emotions across modalities
            normalized_emotions = self._normalize_emotions({
                'face': face_data,
                'speech': speech_data,
                'text': text_data,
                'mental_state': mental_data
            })
            
            # Calculate weighted fusion
            fused_scores = self._calculate_weighted_fusion(normalized_emotions)
            
            # Determine dominant emotion and confidence
            dominant_emotion = max(fused_scores, key=fused_scores.get)
            dominant_confidence = fused_scores[dominant_emotion]
            
            # Calculate emotional coherence across modalities
            coherence_score = self._calculate_coherence(normalized_emotions)
            
            # Generate comprehensive assessment
            fusion_result = {
                'fused_emotion': dominant_emotion,
                'confidence': dominant_confidence,
                'emotion_scores': fused_scores,
                'coherence': coherence_score,
                'modality_contributions': self._get_modality_contributions(normalized_emotions),
                'reliability_score': self._calculate_reliability(multimodal_data),
                'timestamp': datetime.now().isoformat(),
                'raw_modalities': multimodal_data
            }
            
            logger.info(f"Emotion fusion complete: {dominant_emotion} ({dominant_confidence:.2f})")
            return fusion_result
            
        except Exception as e:
            logger.error(f"Emotion fusion failed: {e}")
            return self._get_fallback_fusion()
    
    def _normalize_emotions(self, modality_data: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Normalize emotion labels and scores across modalities"""
        normalized = {}
        
        for modality, data in modality_data.items():
            if not data or not isinstance(data, dict):
                continue
                
            modality_emotions = {}
            
            # Handle different data formats
            if 'emotion_scores' in data:
                # Multi-emotion format (like face/speech)
                emotion_scores = data['emotion_scores']
                for emotion, score in emotion_scores.items():
                    standardized_emotion = self._standardize_emotion_label(emotion)
                    if standardized_emotion:
                        modality_emotions[standardized_emotion] = float(score) / 100.0
                        
            elif 'emotion' in data and 'confidence' in data:
                # Single emotion format (like text)
                emotion = data['emotion']
                confidence = data['confidence']
                standardized_emotion = self._standardize_emotion_label(emotion)
                if standardized_emotion:
                    modality_emotions[standardized_emotion] = float(confidence)
                    
            elif 'mental_state' in data:
                # Mental state format
                mental_state = data['mental_state']
                confidence = data.get('confidence', 0.5)
                # Map mental states to emotions
                emotion = self._mental_state_to_emotion(mental_state)
                if emotion:
                    modality_emotions[emotion] = float(confidence)
            
            # Ensure all base emotions are present
            for base_emotion in ['happy', 'sad', 'angry', 'fear', 'neutral', 'surprise', 'disgust']:
                if base_emotion not in modality_emotions:
                    modality_emotions[base_emotion] = 0.0
            
            normalized[modality] = modality_emotions
        
        return normalized
    
    def _standardize_emotion_label(self, emotion: str) -> Optional[str]:
        """Standardize emotion labels across different models"""
        emotion_lower = emotion.lower()
        
        for standard_emotion, variants in self.emotion_mapping.items():
            if emotion_lower in variants or emotion_lower == standard_emotion:
                return standard_emotion
        
        return None
    
    def _mental_state_to_emotion(self, mental_state: str) -> Optional[str]:
        """Map mental state assessments to emotions"""
        mental_state_lower = mental_state.lower()
        
        if 'normal' in mental_state_lower or 'stable' in mental_state_lower:
            return 'neutral'
        elif 'concern' in mental_state_lower or 'attention' in mental_state_lower:
            return 'sad'
        elif 'unclear' in mental_state_lower:
            return 'neutral'
        
        return 'neutral'
    
    def _calculate_weighted_fusion(self, normalized_emotions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate weighted fusion of emotion scores"""
        fused_scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0, 
                       'neutral': 0.0, 'surprise': 0.0, 'disgust': 0.0}
        
        total_weight = 0.0
        
        for modality, emotions in normalized_emotions.items():
            weight = self.modality_weights.get(modality, 0.1)
            total_weight += weight
            
            for emotion, score in emotions.items():
                fused_scores[emotion] += score * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for emotion in fused_scores:
                fused_scores[emotion] /= total_weight
        
        return fused_scores
    
    def _calculate_coherence(self, normalized_emotions: Dict[str, Dict[str, float]]) -> float:
        """Calculate coherence score across modalities"""
        if len(normalized_emotions) < 2:
            return 0.5
        
        modality_dominants = []
        for modality, emotions in normalized_emotions.items():
            if emotions:
                dominant = max(emotions, key=emotions.get)
                modality_dominants.append(dominant)
        
        if not modality_dominants:
            return 0.5
        
        # Calculate agreement percentage
        most_common = max(set(modality_dominants), key=modality_dominants.count)
        agreement = modality_dominants.count(most_common) / len(modality_dominants)
        
        return agreement
    
    def _get_modality_contributions(self, normalized_emotions: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Get dominant emotion from each modality"""
        contributions = {}
        
        for modality, emotions in normalized_emotions.items():
            if emotions:
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]
                contributions[modality] = f"{dominant} ({confidence:.2f})"
            else:
                contributions[modality] = "no_data"
        
        return contributions
    
    def _calculate_reliability(self, multimodal_data: Dict[str, Any]) -> float:
        """Calculate overall reliability of the assessment"""
        reliability_factors = []
        
        # Check face emotion reliability
        face_data = multimodal_data.get('face_emotion', {})
        if face_data and face_data.get('confidence', 0) > 50:
            reliability_factors.append(0.8)
        
        # Check speech emotion reliability
        speech_data = multimodal_data.get('speech_emotion', {})
        if speech_data and speech_data.get('confidence', 0) > 40:
            reliability_factors.append(0.7)
        
        # Check text availability
        text_data = multimodal_data.get('text_emotion', {})
        transcribed_text = multimodal_data.get('transcribed_text', '')
        if text_data and len(transcribed_text) > 10:
            reliability_factors.append(0.6)
        
        return np.mean(reliability_factors) if reliability_factors else 0.3
    
    def _get_fallback_fusion(self) -> Dict[str, Any]:
        """Fallback fusion result when processing fails"""
        return {
            'fused_emotion': 'neutral',
            'confidence': 0.5,
            'emotion_scores': {'neutral': 0.5, 'happy': 0.1, 'sad': 0.1, 'angry': 0.1, 
                              'fear': 0.1, 'surprise': 0.05, 'disgust': 0.05},
            'coherence': 0.5,
            'modality_contributions': {},
            'reliability_score': 0.3,
            'timestamp': datetime.now().isoformat(),
            'error': 'Fusion processing failed'
        }