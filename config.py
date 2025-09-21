import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for EmotionSense-AI system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config = self._load_default_config()
        
        # Load from file if exists
        if Path(self.config_file).exists():
            self._load_config_file()
        
        # Override with environment variables
        self._load_environment_variables()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            # API Configuration
            'openai_api_key': '',
            'gpt_model': 'gpt-4',
            
            # Model Paths
            'stt_model_path': 'voicemodels/whisper',
            'text_model_path': 'voicemodels/text_emotion', 
            'mental_model_path': 'textmodels/mental',
            
            # Performance Settings
            'max_workers': 4,
            'request_timeout': 30,
            'model_timeout': 15,
            
            # Pipeline Settings
            'pipeline_stages': {
                'face_analysis': {'timeout': 5.0, 'priority': 2},
                'speech_analysis': {'timeout': 6.0, 'priority': 2},
                'transcription': {'timeout': 15.0, 'priority': 3},  # Increased from 8.0 to 15.0
                'text_analysis': {'timeout': 3.0, 'priority': 4},
                'mental_analysis': {'timeout': 3.0, 'priority': 4},
                'fusion': {'timeout': 2.0, 'priority': 5},
                'therapeutic_response': {'timeout': 15.0, 'priority': 6}
            },
            
            # Safety Configuration
            'safety_enabled': True,
            'crisis_detection_enabled': True,
            'intervention_threshold': 'medium',
            
            # Logging Configuration
            'log_level': 'INFO',
            'log_directory': 'logs',
            'log_rotation_size': 10485760,  # 10MB
            'log_backup_count': 5,
            
            # Input Validation
            'validation_rules': {
                'video': {
                    'min_width': 64,
                    'min_height': 64,
                    'max_width': 1920,
                    'max_height': 1080
                },
                'audio': {
                    'min_duration': 0.1,
                    'max_duration': 30.0,
                    'max_file_size': 52428800  # 50MB
                },
                'text': {
                    'min_length': 1,
                    'max_length': 5000
                }
            },
            
            # Performance Profiling
            'profiling_enabled': True,
            'profiling_max_records': 1000,
            
            # API Server Settings
            'api_host': '0.0.0.0',
            'api_port': 8000,
            'api_workers': 1,
            
            # UI Settings
            'ui_host': '0.0.0.0',
            'ui_port': 8501,
            
            # Session Management
            'session_timeout_hours': 24,
            'max_concurrent_sessions': 100,
            
            # Model-specific Settings
            'face_emotion': {
                'analysis_interval': 0.3,
                'confidence_threshold': 0.5
            },
            'speech_emotion': {
                'analysis_interval': 1.0,
                'confidence_threshold': 0.4,
                'chunk_duration': 3.0
            },
            'text_emotion': {
                'min_text_length': 3,
                'confidence_threshold': 0.3
            },
            'mental_state': {
                'min_text_length': 3,
                'confidence_threshold': 0.5
            },
            
            # Therapeutic Response Settings
            'therapeutic_response': {
                'max_retries': 3,
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
    
    def _load_config_file(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        
        # Map environment variables to config keys
        env_mapping = {
            'EMOTIONSENSE_OPENAI_API_KEY': 'openai_api_key',
            'EMOTIONSENSE_GPT_MODEL': 'gpt_model',
            'EMOTIONSENSE_MAX_WORKERS': 'max_workers',
            'EMOTIONSENSE_LOG_LEVEL': 'log_level',
            'EMOTIONSENSE_API_HOST': 'api_host',
            'EMOTIONSENSE_API_PORT': 'api_port',
            'EMOTIONSENSE_UI_PORT': 'ui_port'
        }
        
        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Type conversion
                if config_key in ['max_workers', 'api_port', 'ui_port']:
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {env_value}")
                        continue
                
                self.config[config_key] = env_value
                logger.info(f"Config overridden by environment: {config_key}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return issues"""
        issues = {'errors': [], 'warnings': []}
        
        # Check required fields
        if not self.config.get('openai_api_key'):
            issues['errors'].append("OpenAI API key is required")
        
        # Check model paths exist
        model_paths = ['stt_model_path', 'text_model_path', 'mental_model_path']
        for path_key in model_paths:
            path = self.config.get(path_key)
            if path and not Path(path).exists():
                issues['warnings'].append(f"Model path does not exist: {path}")
        
        # Check numeric values
        numeric_fields = ['max_workers', 'api_port', 'ui_port']
        for field in numeric_fields:
            value = self.config.get(field)
            if not isinstance(value, int) or value <= 0:
                issues['errors'].append(f"Invalid numeric value for {field}: {value}")
        
        return issues
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.config.get(model_name, {})
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return os.getenv('EMOTIONSENSE_ENV', 'production').lower() == 'development'
    
    def get_api_url(self) -> str:
        """Get API base URL"""
        host = self.config.get('api_host', '0.0.0.0')
        port = self.config.get('api_port', 8000)
        
        # Use localhost for 0.0.0.0
        if host == '0.0.0.0':
            host = 'localhost'
        
        return f"http://{host}:{port}"
    
    def get_ui_url(self) -> str:
        """Get UI base URL"""
        host = self.config.get('ui_host', '0.0.0.0')
        port = self.config.get('ui_port', 8501)
        
        # Use localhost for 0.0.0.0
        if host == '0.0.0.0':
            host = 'localhost'
        
        return f"http://{host}:{port}"
    

















    