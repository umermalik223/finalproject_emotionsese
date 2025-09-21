import asyncio
import logging
import time
import os
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime

from models.face_emotion import FastEmotionAnalyzer
from models.speech_emotion import OptimizedSERAnalyzer
from models.speech_to_text import SpeechToTextAnalyzer
from models.text_emotion_analyzer import TextEmotionAnalyzer
from models.mental_state_analyzer import MentalStateAnalyzer
from models.therapeutic_response_generator import TherapeuticResponseGenerator

from core.emotion_fusion import EmotionFusionEngine
from core.pipeline_manager import PipelineManager
from utils.input_validator import InputValidator
from utils.safety_checker import SafetyChecker
from utils.logger import EmotionLogger
from utils.profiler import PerformanceProfiler

logger = logging.getLogger(__name__)

class EmotionSenseOrchestrator:
    """Main orchestrator for EmotionSense-AI therapeutic system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False
        
        # Core components
        self.input_validator = InputValidator()
        self.safety_checker = SafetyChecker()
        self.emotion_logger = EmotionLogger()
        self.performance_profiler = PerformanceProfiler()
        
        # Model instances
        self.models = {}
        
        # Processing components
        self.emotion_fusion = EmotionFusionEngine()
        self.pipeline_manager = PipelineManager(max_workers=config.get('max_workers', 4))
        
        # Session management
        self.current_session = None
        self.session_data = {}
    
    async def initialize(self) -> bool:
        """Initialize all models and components"""
        logger.info("Initializing EmotionSense-AI Orchestrator...")
        
        try:
            # Initialize models in parallel where possible
            await self._initialize_models()
            
            # Verify safety systems
            self.safety_checker.initialize()
            
            # Setup logging
            self.emotion_logger.setup_logging(self.config.get('log_level', 'INFO'))
            
            self.initialized = True
            logger.info("Orchestrator initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            return False
    
    async def _initialize_models(self):
        """Initialize all AI models with error handling"""
        
        initialization_tasks = []
        
        # Face emotion analyzer
        self.models['face_analyzer'] = FastEmotionAnalyzer()
        
        # Speech emotion analyzer  
        self.models['speech_analyzer'] = OptimizedSERAnalyzer()
        
        # Speech-to-text analyzer
        stt_model_path = self.config.get('stt_model_path', 'voicemodels/whisper')
        self.models['stt_analyzer'] = SpeechToTextAnalyzer(stt_model_path)
        
        # Text emotion analyzer
        text_model_path = self.config.get('text_model_path', 'voicemodels/text_emotion')
        self.models['text_analyzer'] = TextEmotionAnalyzer(text_model_path)
        
        # Mental state analyzer
        mental_model_path = self.config.get('mental_model_path', 'textmodels/mental')
        self.models['mental_analyzer'] = MentalStateAnalyzer(mental_model_path)
        
        # Therapeutic response generator (optional)
        openai_api_key = self.config.get('openai_api_key')
        if openai_api_key:
            try:
                self.models['therapeutic_generator'] = TherapeuticResponseGenerator(
                    api_key=openai_api_key,
                    model=self.config.get('gpt_model', 'gpt-4')
                )
                logger.info("Therapeutic response generator loaded")
            except Exception as e:
                logger.warning(f"Failed to load therapeutic generator: {e}")
        else:
            logger.info("OpenAI API key not provided - therapeutic responses disabled")
        
        # Add emotion fusion to models dictionary
        self.models['emotion_fusion'] = self.emotion_fusion
        
        # Load models that need explicit loading
        models_to_load = ['stt_analyzer', 'text_analyzer', 'mental_analyzer']
        for model_name in models_to_load:
            if self.models[model_name] and hasattr(self.models[model_name], 'load_model'):
                success = self.models[model_name].load_model()
                if not success:
                    logger.warning(f"Failed to load {model_name}")
        
        logger.info("All models initialized")
    
    async def process_multimodal_input(
        self, 
        video_frame=None, 
        audio_chunk=None, 
        audio_file=None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main processing function for multimodal therapeutic analysis"""
        
        if not self.initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        process_start = time.time()
        
        try:
            # Start session if needed
            if session_id and session_id != self.current_session:
                await self._start_new_session(session_id)
            
            # Validate inputs
            validation_result = self.input_validator.validate_multimodal_input(
                video_frame=video_frame,
                audio_chunk=audio_chunk,
                audio_file=audio_file
            )
            
            if not validation_result['valid']:
                return self._create_error_response(
                    "Input validation failed", 
                    validation_result['errors']
                )
            
            # Prepare input data
            input_data = self._prepare_input_data(video_frame, audio_chunk, audio_file)
            
            # Execute parallel processing pipeline
            pipeline_results = await self.pipeline_manager.execute_parallel_pipeline(
                input_data, self.models
            )
            
            # Safety check before therapeutic response
            safety_result = await self._perform_safety_check(pipeline_results)
            
            if not safety_result['safe']:
                return self._create_safety_response(safety_result)
            
            # Log session data
            if session_id:
                self._update_session_data(session_id, pipeline_results)
            
            # Create final response
            processing_time = time.time() - process_start
            
            # Format results for frontend
            formatted_results = self._format_response_for_frontend(pipeline_results)
            logger.info(f"Formatted results: {formatted_results}")
            
            response = {
                'success': True,
                **formatted_results,  # Spread formatted results at top level
                'processing_time': processing_time,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'safety_status': safety_result
            }
            
            
            logger.info(f"Multimodal processing completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            
            
            return self._create_error_response("Processing failed", str(e))
    
    def _prepare_input_data(self, video_frame, audio_chunk, audio_file) -> Dict[str, Any]:
        """Prepare input data for pipeline processing"""
        input_data = {}
        
        if video_frame is not None:
            input_data['video_frame'] = video_frame
        
        if audio_chunk is not None:
            input_data['audio_chunk'] = audio_chunk
        
        if audio_file is not None:
            input_data['audio_file'] = audio_file
        elif audio_chunk is not None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            )
            input_data['audio_file'] = temp_file.name
        
        return input_data
    
    async def _perform_safety_check(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive safety check before therapeutic response"""
        
        # Extract key emotional data
        fusion_result = pipeline_results.get('emotion_fusion', {})
        transcribed_text = pipeline_results.get('transcribed_text', '')
        mental_state = pipeline_results.get('mental_state', {})
        
        # Perform safety analysis
        safety_analysis = self.safety_checker.analyze_safety(
            fused_emotion=fusion_result.get('fused_emotion'),
            confidence=fusion_result.get('confidence', 0.0),
            transcribed_text=transcribed_text,
            mental_state=mental_state.get('mental_state'),
            emotion_coherence=fusion_result.get('coherence', 0.0)
        )
        
        return safety_analysis
    
    async def _start_new_session(self, session_id: str):
        """Start new therapeutic session"""
        self.current_session = session_id
        self.session_data[session_id] = {
            'start_time': datetime.now().isoformat(),
            'interactions': [],
            'emotional_progression': [],
            'safety_flags': []
        }
        
        logger.info(f"Started new session: {session_id}")
    
    def _update_session_data(self, session_id: str, results: Dict[str, Any]):
        """Update session data with latest results"""
        if session_id in self.session_data:
            self.session_data[session_id]['interactions'].append({
                'timestamp': datetime.now().isoformat(),
                'results': results
            })
            
            # Track emotional progression
            fusion_result = results.get('emotion_fusion', {})
            if fusion_result:
                self.session_data[session_id]['emotional_progression'].append({
                    'timestamp': datetime.now().isoformat(),
                    'emotion': fusion_result.get('fused_emotion'),
                    'confidence': fusion_result.get('confidence')
                })
    
    def _create_error_response(self, error_message: str, details: Any) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'fallback_response': {
                'therapeutic_response': {
                    'empathetic_response': "I'm here to support you. Let's try again.",
                    'calming_techniques': [
                        "Take a moment to breathe deeply",
                        "Focus on the present moment",
                        "Remember that it's okay to pause"
                    ],
                    'emotional_validation': "Your feelings are important",
                    'severity_assessment': 'low'
                }
            }
        }
    
    def _create_safety_response(self, safety_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create response for safety concerns"""
        return {
            'success': True,
            'safety_intervention': True,
            'message': safety_result.get('message', 'Safety protocols activated'),
            'recommended_actions': safety_result.get('recommended_actions', []),
            'crisis_resources': safety_result.get('crisis_resources', []),
            'severity_level': safety_result.get('severity_level', 'low'),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        if session_id not in self.session_data:
            return {'error': 'Session not found'}
        
        session = self.session_data[session_id]
        
        # Analyze emotional progression
        emotions = [e['emotion'] for e in session['emotional_progression']]
        emotion_changes = len(set(emotions)) if emotions else 0
        
        return {
            'session_id': session_id,
            'duration': session['start_time'],
            'total_interactions': len(session['interactions']),
            'emotional_progression': session['emotional_progression'],
            'emotion_changes': emotion_changes,
            'safety_flags': session['safety_flags'],
            'performance_metrics': self.performance_profiler.get_session_metrics()
        }
    
    async def analyze_text_only(self, text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text input only for emotion detection"""
        
        if not self.initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        process_start = time.time()
        
        try:
            # Start session if needed
            if session_id and session_id != self.current_session:
                await self._start_new_session(session_id)
            
            # Analyze text emotion
            text_emotion_result = {}
            mental_state_result = {}
            
            if 'text_analyzer' in self.models:
                text_emotion_result = self.models['text_analyzer'].analyze_text_emotion(text)
            
            if 'mental_analyzer' in self.models and self.models['mental_analyzer'] is not None:
                mental_state_result = self.models['mental_analyzer'].analyze_mental_state(text)
            
            # Generate therapeutic response
            therapeutic_response = {}
            if 'therapeutic_generator' in self.models and self.models['therapeutic_generator'] is not None:
                try:
                    # Create user data for therapeutic analysis
                    user_data = {
                        'text_emotion': text_emotion_result,
                        'mental_state': mental_state_result,
                        'transcribed_text': text,
                        'face_emotion': {'emotion': 'neutral', 'confidence': 0.0},  # No video
                        'speech_emotion': {'emotion': 'neutral', 'confidence': 0.0},  # No audio
                        'emotion_fusion': {
                            'fused_emotion': text_emotion_result.get('emotion', 'neutral'),
                            'confidence': text_emotion_result.get('confidence', 0.0)
                        }
                    }
                    
                    therapeutic_response = await self.models['therapeutic_generator'].generate_therapeutic_response(user_data)
                    logger.info("✅ Therapeutic response generated for text analysis")
                except Exception as e:
                    logger.error(f"Failed to generate therapeutic response: {e}")
                    therapeutic_response = {'success': False, 'error': str(e)}
            
            # Create response in frontend-expected format
            processing_time = time.time() - process_start
            
            # Helper function to convert numpy types
            def convert_to_python_type(value):
                if hasattr(value, 'item'):  # numpy scalar
                    return value.item()
                elif hasattr(value, 'tolist'):  # numpy array
                    return value.tolist()
                return value
            
            response = {
                'success': True,
                'text_emotion': {
                    'emotion': str(text_emotion_result.get('emotion', 'neutral')),
                    'confidence': convert_to_python_type(text_emotion_result.get('confidence', 0.0))
                },
                'mental_state': {
                    'emotion': str(mental_state_result.get('mental_state', 'unknown')),
                    'confidence': convert_to_python_type(mental_state_result.get('confidence', 0.0))
                },
                'processing_time': processing_time,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add therapeutic response if generated successfully
            if therapeutic_response and therapeutic_response.get('success', False):
                response['therapeutic_response'] = therapeutic_response
            
            # Log session data
            if session_id:
                self._update_session_data(session_id, {
                    'text_emotion': text_emotion_result,
                    'mental_state': mental_state_result,
                    'input_text': text
                })
            
            logger.info(f"Text analysis completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_error_response("Text analysis failed", str(e))
    
    def _format_response_for_frontend(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format pipeline results for frontend consumption"""
        
        formatted_response = {}
        
        # Helper function to convert numpy types to Python types
        def convert_to_python_type(value):
            if hasattr(value, 'item'):  # numpy scalar
                return value.item()
            elif hasattr(value, 'tolist'):  # numpy array
                return value.tolist()
            return value
        
        # Facial emotion
        if 'face_emotion' in pipeline_results:
            face_result = pipeline_results['face_emotion']
            formatted_response['facial_emotion'] = {
                'emotion': str(face_result.get('emotion', 'neutral')),
                'confidence': convert_to_python_type(face_result.get('confidence', 0.0))
            }
        
        # Speech emotion
        if 'speech_emotion' in pipeline_results:
            speech_result = pipeline_results['speech_emotion']
            formatted_response['speech_emotion'] = {
                'emotion': str(speech_result.get('emotion', 'neutral')),
                'confidence': convert_to_python_type(speech_result.get('confidence', 0.0))
            }
        
        # Speech to text - Enhanced with robust fallback handling
        if 'speech_to_text' in pipeline_results:
            stt_result = pipeline_results['speech_to_text']
            formatted_response['speech_to_text'] = {
                'text': str(stt_result.get('text', '')),
                'confidence': convert_to_python_type(stt_result.get('confidence', 0.0))
            }
            logger.info(f"✅ Speech-to-text included in response: '{stt_result.get('text', '')}'")
        else:
            # Fallback: Check if there was transcribed_text available but STT result got lost due to timeout
            transcribed_text = pipeline_results.get('transcribed_text', '')
            if transcribed_text and len(transcribed_text.strip()) > 0:
                # Use the transcribed text with estimated confidence
                formatted_response['speech_to_text'] = {
                    'text': str(transcribed_text),
                    'confidence': 0.8  # High confidence since text was successfully transcribed
                }
                logger.info(f"🔄 Speech-to-text recovered from transcribed_text: '{transcribed_text}'")
            else:
                # Complete fallback when no audio was processed
                formatted_response['speech_to_text'] = {
                    'text': '',
                    'confidence': 0.0
                }
                logger.warning("⚠️ Speech-to-text fallback: no audio processed")
        
        # Text emotion
        if 'text_emotion' in pipeline_results:
            text_result = pipeline_results['text_emotion']
            formatted_response['text_emotion'] = {
                'emotion': str(text_result.get('emotion', 'neutral')),
                'confidence': convert_to_python_type(text_result.get('confidence', 0.0))
            }
        
        # Mental state
        if 'mental_state' in pipeline_results:
            mental_result = pipeline_results['mental_state']
            formatted_response['mental_state'] = {
                'emotion': str(mental_result.get('mental_state', 'unknown')),
                'confidence': convert_to_python_type(mental_result.get('confidence', 0.0))
            }
        
        # Fused emotion
        if 'emotion_fusion' in pipeline_results:
            fusion_result = pipeline_results['emotion_fusion']
            formatted_response['fused_emotion'] = {
                'emotion': str(fusion_result.get('fused_emotion', 'neutral')),
                'confidence': convert_to_python_type(fusion_result.get('confidence', 0.0))
            }
        
        # Therapeutic response
        if 'therapeutic_response' in pipeline_results:
            therapeutic_result = pipeline_results['therapeutic_response']
            if therapeutic_result and therapeutic_result.get('success', False):
                formatted_response['therapeutic_response'] = therapeutic_result
        
        return formatted_response

    async def shutdown(self):
        """Clean shutdown of orchestrator"""
        logger.info("Shutting down orchestrator...")
        
        # Save session data
        for session_id, session_data in self.session_data.items():
            await self.emotion_logger.save_session(session_id, session_data)
        
        # Cleanup models
        for model in self.models.values():
            if hasattr(model, 'cleanup'):
                model.cleanup()
        
        # Cleanup pipeline manager
        if hasattr(self.pipeline_manager, '__del__'):
            self.pipeline_manager.__del__()
        
        logger.info("Orchestrator shutdown complete")