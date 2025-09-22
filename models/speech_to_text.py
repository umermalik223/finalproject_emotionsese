import os
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SpeechToTextAnalyzer:
   
    def __init__(self, model_path="voicemodels/whisper"):
        self.model_path = Path(model_path)
        self.loaded = False
        
        # Always use local transformers model first (it's faster and already downloaded)
        self.use_whisper_api = False
        self.processor = None
        self.model = None
        self.device = None
        
        logger.info(f"Will use local Whisper model from: {self.model_path}")
    
    def load_model(self) -> bool:
        """Load local Whisper model"""
        logger.info("Loading local Whisper speech-to-text model...")
        
        try:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            # Use the local model path directly
            model_path_str = str(self.model_path.absolute())
            logger.info(f"Loading local Whisper model from: {model_path_str}")
            
            # Load local processor and model
            self.processor = WhisperProcessor.from_pretrained(model_path_str)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path_str)
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Local Whisper model loaded successfully on {self.device}")
            logger.info(f"Model config: {self.model.config.architectures}")
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local Whisper model: {str(e)}")
            logger.debug(traceback.format_exc())
            self.loaded = False
            return False
    
    def transcribe(self, audio_file) -> Dict[str, Any]:
        """Transcribe audio file to text using local Whisper model"""
        if not self.loaded:
            return {
                'text': '',
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Validate audio file
            if not os.path.exists(audio_file):
                return {
                    'text': '',
                    'confidence': 0.0,
                    'error': 'Audio file not found'
                }
            
            import librosa
            import numpy as np
            import torch
            
            logger.info(f"Transcribing audio file: {audio_file}")
            
            # Load and preprocess audio
            audio_input, sr = librosa.load(audio_file, sr=16000, mono=True)
            logger.info(f"Loaded audio: {len(audio_input)} samples at {sr}Hz")
            
            # Check audio quality
            audio_rms = np.sqrt(np.mean(audio_input**2))
            audio_max = np.max(np.abs(audio_input))
            logger.info(f"Audio quality - RMS: {audio_rms:.6f}, Max: {audio_max:.6f}")
            
            # Check for minimum audio length
            if len(audio_input) < 1600:  # Less than 0.1 seconds 
                logger.warning("Audio too short for transcription")
                return {
                    'text': '',
                    'confidence': 0.0,
                    'error': 'Audio too short for reliable transcription'
                }
            
            # Check if audio has meaningful content
            silence_threshold = 0.001
            if audio_rms < silence_threshold:
                logger.warning(f"Audio appears to be mostly silence (RMS: {audio_rms:.6f})")
                return {
                    'text': '',
                    'confidence': 0.0,
                    'error': 'Audio contains mostly silence'
                }
            
            # Process audio with local Whisper model
            logger.info("Processing with local Whisper model...")
            inputs = self.processor(
                audio_input,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription (using working version settings)
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    max_length=224,
                    num_beams=2,
                    early_stopping=True,
                    language="en",
                    task="transcribe",
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True,
                    no_repeat_ngram_size=3,
                    forced_decoder_ids=self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()
            
            logger.info(f"Local Whisper transcription: '{transcription}' (length: {len(transcription)})")
            
            # Calculate confidence based on transcription quality
            if len(transcription) > 0:
                # Higher confidence for longer, more complete transcriptions
                confidence = min(0.95, 0.7 + (len(transcription) / 100.0))
            else:
                confidence = 0.0
            
            return {
                'text': transcription,
                'confidence': confidence,
                'length': len(transcription),
                'audio_duration': len(audio_input) / sr,
                'model_type': 'local_whisper_transformers',
                'audio_quality': {
                    'rms': audio_rms,
                    'max_amplitude': audio_max
                }
            }
            
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded
