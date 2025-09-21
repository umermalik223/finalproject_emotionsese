import cv2
import numpy as np
import librosa
import os
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Comprehensive input validation for multimodal data"""
    
    def __init__(self):
        self.validation_rules = {
            'video': {
                'min_width': 64,
                'min_height': 64,
                'max_width': 1920,
                'max_height': 1080,
                'supported_channels': [1, 3, 4]
            },
            'audio': {
                'min_duration': 0.1,  # seconds
                'max_duration': 30.0,  # seconds
                'min_sample_rate': 8000,
                'max_sample_rate': 48000,
                'max_file_size': 50 * 1024 * 1024  # 50MB
            },
            'text': {
                'min_length': 1,
                'max_length': 5000,
                'encoding': 'utf-8'
            }
        }
    
    def validate_multimodal_input(
        self, 
        video_frame=None, 
        audio_chunk=None, 
        audio_file=None,
        text=None
    ) -> Dict[str, Any]:
        """Validate all multimodal inputs comprehensively"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'input_quality': {}
        }
        
        try:
            # Validate video frame
            if video_frame is not None:
                video_validation = self._validate_video_frame(video_frame)
                validation_result['input_quality']['video'] = video_validation
                if not video_validation['valid']:
                    validation_result['valid'] = False
                    validation_result['errors'].extend(video_validation['errors'])
            
            # Validate audio chunk
            if audio_chunk is not None:
                audio_validation = self._validate_audio_chunk(audio_chunk)
                validation_result['input_quality']['audio_chunk'] = audio_validation
                if not audio_validation['valid']:
                    validation_result['valid'] = False
                    validation_result['errors'].extend(audio_validation['errors'])
            
            # Validate audio file
            if audio_file is not None:
                file_validation = self._validate_audio_file(audio_file)
                validation_result['input_quality']['audio_file'] = file_validation
                if not file_validation['valid']:
                    validation_result['valid'] = False
                    validation_result['errors'].extend(file_validation['errors'])
            
            # Validate text
            if text is not None:
                text_validation = self._validate_text(text)
                validation_result['input_quality']['text'] = text_validation
                if not text_validation['valid']:
                    validation_result['valid'] = False
                    validation_result['errors'].extend(text_validation['errors'])
            
            # Check if at least one valid input provided
            if not any([video_frame is not None, audio_chunk is not None, 
                       audio_file is not None, text is not None]):
                validation_result['valid'] = False
                validation_result['errors'].append("No valid input provided")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Input validation failed: {e}")
        
        return validation_result
    
    def _validate_video_frame(self, frame) -> Dict[str, Any]:
        """Validate video frame for face emotion analysis"""
        try:
            if not isinstance(frame, np.ndarray):
                return {
                    'valid': False,
                    'errors': ['Video frame must be numpy array'],
                    'quality_score': 0.0
                }
            
            # Check dimensions
            if len(frame.shape) not in [2, 3]:
                return {
                    'valid': False,
                    'errors': ['Video frame must be 2D or 3D array'],
                    'quality_score': 0.0
                }
            
            height, width = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1
            
            errors = []
            warnings = []
            
            # Validate dimensions
            if width < self.validation_rules['video']['min_width']:
                errors.append(f"Frame width too small: {width}")
            if height < self.validation_rules['video']['min_height']:
                errors.append(f"Frame height too small: {height}")
            if width > self.validation_rules['video']['max_width']:
                warnings.append(f"Frame width very large: {width}")
            if height > self.validation_rules['video']['max_height']:
                warnings.append(f"Frame height very large: {height}")
            
            # Validate channels
            if channels not in self.validation_rules['video']['supported_channels']:
                errors.append(f"Unsupported channel count: {channels}")
            
            # Check data type and range
            if frame.dtype not in [np.uint8, np.float32, np.float64]:
                warnings.append(f"Unusual data type: {frame.dtype}")
            
            # Calculate quality metrics
            quality_score = self._calculate_video_quality(frame)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'quality_score': quality_score,
                'dimensions': (width, height, channels),
                'dtype': str(frame.dtype)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Video validation error: {str(e)}"],
                'quality_score': 0.0
            }
    
    def _validate_audio_chunk(self, audio_chunk) -> Dict[str, Any]:
        """Validate audio chunk for speech analysis"""
        try:
            if not isinstance(audio_chunk, np.ndarray):
                return {
                    'valid': False,
                    'errors': ['Audio chunk must be numpy array'],
                    'quality_score': 0.0
                }
            
            # Ensure 1D array
            if len(audio_chunk.shape) > 2:
                return {
                    'valid': False,
                    'errors': ['Audio chunk must be 1D or 2D array'],
                    'quality_score': 0.0
                }
            
            # Flatten if 2D
            if len(audio_chunk.shape) == 2:
                audio_chunk = audio_chunk.flatten()
            
            errors = []
            warnings = []
            
            # Check duration (assuming 16kHz)
            duration = len(audio_chunk) / 16000
            if duration < self.validation_rules['audio']['min_duration']:
                errors.append(f"Audio too short: {duration:.2f}s")
            if duration > self.validation_rules['audio']['max_duration']:
                warnings.append(f"Audio very long: {duration:.2f}s")
            
            # Check for silence
            max_amplitude = np.max(np.abs(audio_chunk))
            if max_amplitude < 0.001:
                warnings.append("Audio appears to be silent")
            
            # Check for clipping
            if max_amplitude > 0.95:
                warnings.append("Audio may be clipped")
            
            # Calculate quality score
            quality_score = self._calculate_audio_quality(audio_chunk)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'quality_score': quality_score,
                'duration': duration,
                'max_amplitude': float(max_amplitude),
                'sample_count': len(audio_chunk)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Audio chunk validation error: {str(e)}"],
                'quality_score': 0.0
            }
    
    def _validate_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """Validate audio file for transcription"""
        try:
            if not os.path.exists(audio_file_path):
                return {
                    'valid': False,
                    'errors': ['Audio file does not exist'],
                    'quality_score': 0.0
                }
            
            # Check file size
            file_size = os.path.getsize(audio_file_path)
            if file_size > self.validation_rules['audio']['max_file_size']:
                return {
                    'valid': False,
                    'errors': [f'Audio file too large: {file_size} bytes'],
                    'quality_score': 0.0
                }
            
            # Try to load with librosa
            try:
                audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
            except Exception as e:
                return {
                    'valid': False,
                    'errors': [f'Cannot load audio file: {str(e)}'],
                    'quality_score': 0.0
                }
            
            errors = []
            warnings = []
            
            # Validate sample rate
            if sample_rate < self.validation_rules['audio']['min_sample_rate']:
                warnings.append(f"Low sample rate: {sample_rate}Hz")
            if sample_rate > self.validation_rules['audio']['max_sample_rate']:
                warnings.append(f"Very high sample rate: {sample_rate}Hz")
            
            # Validate duration
            duration = len(audio_data) / sample_rate
            if duration < self.validation_rules['audio']['min_duration']:
                errors.append(f"Audio too short: {duration:.2f}s")
            if duration > self.validation_rules['audio']['max_duration']:
                warnings.append(f"Audio very long: {duration:.2f}s")
            
            # Calculate quality
            quality_score = self._calculate_audio_quality(audio_data)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'quality_score': quality_score,
                'duration': duration,
                'sample_rate': sample_rate,
                'file_size': file_size
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Audio file validation error: {str(e)}"],
                'quality_score': 0.0
            }
    
    def _validate_text(self, text: str) -> Dict[str, Any]:
        """Validate text input"""
        try:
            if not isinstance(text, str):
                return {
                    'valid': False,
                    'errors': ['Text must be string'],
                    'quality_score': 0.0
                }
            
            errors = []
            warnings = []
            
            # Check length
            text_length = len(text.strip())
            if text_length < self.validation_rules['text']['min_length']:
                errors.append(f"Text too short: {text_length} characters")
            if text_length > self.validation_rules['text']['max_length']:
                errors.append(f"Text too long: {text_length} characters")
            
            # Check encoding
            try:
                text.encode(self.validation_rules['text']['encoding'])
            except UnicodeEncodeError:
                warnings.append("Text contains non-UTF-8 characters")
            
            # Calculate quality
            quality_score = self._calculate_text_quality(text)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'quality_score': quality_score,
                'length': text_length,
                'word_count': len(text.split())
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Text validation error: {str(e)}"],
                'quality_score': 0.0
            }
    
    def _calculate_video_quality(self, frame: np.ndarray) -> float:
        """Calculate video quality score (0-1)"""
        try:
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000)
            
            # Calculate brightness distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            brightness_score = 1.0 - np.abs(0.5 - np.mean(gray) / 255)
            
            # Combine scores
            quality_score = (sharpness_score * 0.6 + brightness_score * 0.4)
            return max(0.1, min(1.0, quality_score))
            
        except:
            return 0.5
    
    def _calculate_audio_quality(self, audio_data: np.ndarray) -> float:
        """Calculate audio quality score (0-1)"""
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_data ** 2)
            noise_floor = np.percentile(np.abs(audio_data), 10) ** 2
            snr = signal_power / (noise_floor + 1e-10)
            snr_score = min(1.0, np.log10(snr + 1) / 3)
            
            # Dynamic range
            dynamic_range = np.max(np.abs(audio_data)) - np.min(np.abs(audio_data))
            range_score = min(1.0, dynamic_range)
            
            # Combine scores
            quality_score = (snr_score * 0.7 + range_score * 0.3)
            return max(0.1, min(1.0, quality_score))
            
        except:
            return 0.5
    
    def _calculate_text_quality(self, text: str) -> float:
        """Calculate text quality score (0-1)"""
        try:
            # Word count factor
            word_count = len(text.split())
            word_score = min(1.0, word_count / 50)  # Optimal around 50 words
            
            # Character diversity
            unique_chars = len(set(text.lower()))
            diversity_score = min(1.0, unique_chars / 20)
            
            # Sentence structure (basic check)
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            if sentence_count > 0:
                avg_sentence_length = word_count / sentence_count
                structure_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.5
            else:
                structure_score = 0.5
            
            # Combine scores
            quality_score = (word_score * 0.4 + diversity_score * 0.3 + structure_score * 0.3)
            return max(0.1, min(1.0, quality_score))
            
        except:
            return 0.5