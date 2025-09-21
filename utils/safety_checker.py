import re
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SafetyChecker:
    """Comprehensive safety checker for therapeutic AI system"""
    
    def __init__(self):
        self.crisis_keywords = {
            'suicide': [
                'kill myself', 'end my life', 'want to die', 'suicide', 'suicidal',
                'not worth living', 'better off dead', 'end it all', 'take my life',
                'kill me', 'want to be dead', 'wish i was dead'
            ],
            'self_harm': [
                'cut myself', 'hurt myself', 'self harm', 'self-harm', 'cutting',
                'burn myself', 'hit myself', 'harm myself', 'damage myself'
            ],
            'violence': [
                'kill someone', 'hurt someone', 'violent thoughts', 'want to hurt',
                'kill them', 'murder', 'violent urges', 'harm others'
            ],
            'severe_distress': [
                'can\'t take it', 'overwhelming', 'breaking down', 'falling apart',
                'losing control', 'can\'t cope', 'desperate', 'hopeless'
            ]
        }
        
        self.crisis_resources = {
            'suicide_prevention': [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
            ],
            'mental_health_emergency': [
                "Call 911 for immediate emergency",
                "National Mental Health Emergency: 1-800-662-4357",
                "Crisis Text Line: Text HELLO to 741741"
            ]
        }
        
        self.severity_thresholds = {
            'high': {
                'keywords': ['suicide', 'self_harm', 'violence'],
                'emotion_intensity': 0.8,
                'coherence_threshold': 0.3
            },
            'medium': {
                'keywords': ['severe_distress'],
                'emotion_intensity': 0.6,
                'coherence_threshold': 0.4
            },
            'low': {
                'keywords': [],
                'emotion_intensity': 0.4,
                'coherence_threshold': 0.5
            }
        }
        
        self.prohibited_responses = {
            'medical_advice': [
                'diagnose', 'prescribe', 'medication', 'treatment plan',
                'medical condition', 'disorder', 'illness'
            ],
            'crisis_minimization': [
                'get over it', 'just think positive', 'it\'s not that bad',
                'others have it worse', 'snap out of it'
            ]
        }
        
        self.initialized = False
    
    def initialize(self):
        """Initialize safety checker"""
        logger.info("Initializing safety checker...")
        self.initialized = True
        logger.info("Safety checker initialized")
    
    def analyze_safety(
        self,
        fused_emotion: Optional[str] = None,
        confidence: float = 0.0,
        transcribed_text: str = "",
        mental_state: Optional[str] = None,
        emotion_coherence: float = 0.0
    ) -> Dict[str, Any]:
        """Comprehensive safety analysis"""
        
        if not self.initialized:
            logger.warning("Safety checker not initialized")
        
        try:
            # Initialize safety result
            safety_result = {
                'safe': True,
                'severity_level': 'low',
                'risk_factors': [],
                'protective_factors': [],
                'recommended_actions': [],
                'crisis_resources': [],
                'message': 'No immediate safety concerns detected',
                'requires_intervention': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Analyze text for crisis keywords
            text_analysis = self._analyze_crisis_keywords(transcribed_text)
            
            # Analyze emotional state
            emotion_analysis = self._analyze_emotional_risk(
                fused_emotion, confidence, emotion_coherence
            )
            
            # Analyze mental state indicators
            mental_analysis = self._analyze_mental_state_risk(mental_state)
            
            # Combine analyses
            combined_risk = self._combine_risk_assessments(
                text_analysis, emotion_analysis, mental_analysis
            )
            
            # Determine overall safety status
            safety_result = self._determine_safety_response(
                combined_risk, safety_result
            )
            
            # Log safety assessment
            self._log_safety_assessment(safety_result, transcribed_text)
            
            return safety_result
            
        except Exception as e:
            logger.error(f"Safety analysis failed: {e}")
            return self._get_emergency_fallback()
    
    def _analyze_crisis_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze text for crisis-related keywords"""
        
        text_lower = text.lower()
        detected_keywords = {}
        risk_score = 0.0
        
        for category, keywords in self.crisis_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
                    # Weight keywords by severity
                    if category == 'suicide':
                        risk_score += 1.0
                    elif category == 'self_harm':
                        risk_score += 0.8
                    elif category == 'violence':
                        risk_score += 0.9
                    elif category == 'severe_distress':
                        risk_score += 0.3
            
            if found_keywords:
                detected_keywords[category] = found_keywords
        
        return {
            'detected_keywords': detected_keywords,
            'risk_score': min(risk_score, 3.0),  # Cap at 3.0
            'high_risk': risk_score > 1.0
        }
    
    def _analyze_emotional_risk(
        self,
        emotion: Optional[str],
        confidence: float,
        coherence: float
    ) -> Dict[str, Any]:
        """Analyze emotional indicators for risk"""
        
        risk_score = 0.0
        risk_factors = []
        
        # High-risk emotions
        high_risk_emotions = ['sad', 'angry', 'fear', 'disgust']
        if emotion in high_risk_emotions and confidence > 0.7:
            risk_score += 0.5
            risk_factors.append(f"High confidence {emotion} emotion")
        
        # Very low coherence might indicate distress
        if coherence < 0.3:
            risk_score += 0.3
            risk_factors.append("Low emotional coherence")
        
        # Extreme confidence might indicate crisis
        if confidence > 0.9:
            risk_score += 0.2
            risk_factors.append("Extreme emotional intensity")
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'emotion_concern': risk_score > 0.5
        }
    
    def _analyze_mental_state_risk(self, mental_state: Optional[str]) -> Dict[str, Any]:
        """Analyze mental state indicators"""
        
        risk_score = 0.0
        risk_factors = []
        
        if mental_state:
            mental_lower = mental_state.lower()
            
            if 'requires attention' in mental_lower:
                risk_score += 0.7
                risk_factors.append("Mental state requires attention")
            elif 'concern' in mental_lower:
                risk_score += 0.4
                risk_factors.append("Mental state shows concern")
            elif 'unclear' in mental_lower or 'error' in mental_lower:
                risk_score += 0.1
                risk_factors.append("Mental state unclear")
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'mental_concern': risk_score > 0.5
        }
    
    def _combine_risk_assessments(
        self,
        text_analysis: Dict[str, Any],
        emotion_analysis: Dict[str, Any],
        mental_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all risk assessments"""
        
        total_risk_score = (
            text_analysis['risk_score'] +
            emotion_analysis['risk_score'] +
            mental_analysis['risk_score']
        )
        
        # Combine all risk factors
        all_risk_factors = []
        
        # Add text risk factors (keywords)
        if text_analysis.get('detected_keywords'):
            for category, keywords in text_analysis['detected_keywords'].items():
                all_risk_factors.extend([f"{category}: {kw}" for kw in keywords])
        
        # Add emotion risk factors
        all_risk_factors.extend(emotion_analysis['risk_factors'])
        
        # Add mental state risk factors  
        all_risk_factors.extend(mental_analysis['risk_factors'])
        
        # Determine severity level
        if total_risk_score >= 2.0 or text_analysis['high_risk']:
            severity = 'high'
        elif total_risk_score >= 1.0:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'total_risk_score': total_risk_score,
            'severity_level': severity,
            'risk_factors': all_risk_factors,
            'crisis_detected': text_analysis['high_risk'],
            'requires_intervention': severity in ['high', 'medium']
        }
    
    def _determine_safety_response(
        self,
        combined_risk: Dict[str, Any],
        safety_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine appropriate safety response"""
        
        severity = combined_risk['severity_level']
        safety_result['severity_level'] = severity
        safety_result['risk_factors'] = combined_risk['risk_factors']
        safety_result['requires_intervention'] = combined_risk['requires_intervention']
        
        if severity == 'high':
            safety_result.update({
                'safe': False,
                'message': 'Immediate safety concerns detected. Crisis intervention protocols activated.',
                'recommended_actions': [
                    'Immediate professional intervention required',
                    'Contact crisis services immediately',
                    'Do not leave person alone if possible',
                    'Remove any means of self-harm'
                ],
                'crisis_resources': (
                    self.crisis_resources['suicide_prevention'] +
                    self.crisis_resources['mental_health_emergency']
                ),
                'requires_intervention': True
            })
        
        elif severity == 'medium':
            safety_result.update({
                'safe': True,  # But with close monitoring
                'message': 'Moderate safety concerns detected. Enhanced support recommended.',
                'recommended_actions': [
                    'Encourage professional support',
                    'Monitor closely for escalation',
                    'Provide crisis resources',
                    'Follow up within 24 hours'
                ],
                'crisis_resources': self.crisis_resources['mental_health_emergency'],
                'requires_intervention': True
            })
        
        else:  # low severity
            safety_result.update({
                'safe': True,
                'message': 'No immediate safety concerns. Continue supportive care.',
                'recommended_actions': [
                    'Continue therapeutic support',
                    'Monitor emotional state',
                    'Encourage self-care'
                ]
            })
        
        return safety_result
    
    def _log_safety_assessment(self, safety_result: Dict[str, Any], text: str):
        """Log safety assessment for monitoring"""
        
        if safety_result['severity_level'] in ['high', 'medium']:
            logger.warning(
                f"Safety concern detected - Severity: {safety_result['severity_level']}, "
                f"Intervention required: {safety_result['requires_intervention']}"
            )
            
            # Don't log the actual text for privacy, just indicators
            logger.warning(f"Risk factors: {len(safety_result['risk_factors'])}")
    
    def _get_emergency_fallback(self) -> Dict[str, Any]:
        """Emergency fallback when safety analysis fails"""
        return {
            'safe': False,
            'severity_level': 'high',
            'risk_factors': ['Safety analysis failed'],
            'protective_factors': [],
            'recommended_actions': [
                'Immediate professional consultation required',
                'Safety analysis system error - manual review needed'
            ],
            'crisis_resources': (
                self.crisis_resources['suicide_prevention'] +
                self.crisis_resources['mental_health_emergency']
            ),
            'message': 'Safety system error - immediate professional review required',
            'requires_intervention': True,
            'error': 'Safety analysis failed',
            'timestamp': datetime.now().isoformat()
        }
    
    def check_response_safety(self, therapeutic_response: Dict[str, Any]) -> Dict[str, Any]:
        """Check if therapeutic response is safe to send"""
        
        response_text = therapeutic_response.get('empathetic_response', '') + ' ' + \
                       ' '.join(therapeutic_response.get('calming_techniques', []))
        
        response_lower = response_text.lower()
        
        # Check for prohibited content
        prohibited_found = []
        for category, phrases in self.prohibited_responses.items():
            for phrase in phrases:
                if phrase in response_lower:
                    prohibited_found.append(f"{category}: {phrase}")
        
        if prohibited_found:
            return {
                'safe_to_send': False,
                'issues_found': prohibited_found,
                'recommended_action': 'Use fallback response'
            }
        
        return {
            'safe_to_send': True,
            'issues_found': [],
            'recommended_action': 'Send response'
        }