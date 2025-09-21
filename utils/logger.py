import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
from pathlib import Path

class EmotionLogger:
    """Comprehensive logging system for EmotionSense-AI"""
    
    def __init__(self):
        self.log_directory = Path("logs")
        self.session_directory = Path("sessions")
        self.performance_directory = Path("performance")
        
        # Create directories
        self.log_directory.mkdir(exist_ok=True)
        self.session_directory.mkdir(exist_ok=True)
        self.performance_directory.mkdir(exist_ok=True)
        
        self.loggers = {}
        self.initialized = False
    
    def setup_logging(self, log_level: str = "INFO"):
        """Setup comprehensive logging system"""
        
        # Convert log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Main application logger
        self._setup_logger(
            'main', 
            self.log_directory / 'emotion_sense.log',
            numeric_level,
            detailed_formatter
        )
        
        # Performance logger
        self._setup_logger(
            'performance',
            self.performance_directory / 'performance.log',
            logging.INFO,
            simple_formatter
        )
        
        # Safety logger (always INFO level minimum)
        safety_level = min(numeric_level, logging.INFO)
        self._setup_logger(
            'safety',
            self.log_directory / 'safety.log',
            safety_level,
            detailed_formatter
        )
        
        # Session logger
        self._setup_logger(
            'session',
            self.session_directory / 'sessions.log',
            logging.INFO,
            simple_formatter
        )
        
        # Error logger (always DEBUG)
        self._setup_logger(
            'error',
            self.log_directory / 'errors.log',
            logging.DEBUG,
            detailed_formatter
        )
        
        self.initialized = True
        self.log_info("Logging system initialized")
    
    def _setup_logger(
        self, 
        name: str, 
        log_file: Path, 
        level: int, 
        formatter: logging.Formatter
    ):
        """Setup individual logger with rotation"""
        
        logger = logging.getLogger(f'emotion_sense.{name}')
        logger.setLevel(level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        
        # Console handler for important logs
        if name in ['main', 'safety', 'error']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(max(level, logging.WARNING))
            logger.addHandler(console_handler)
        
        self.loggers[name] = logger
    
    def log_info(self, message: str, extra_data: Optional[Dict] = None):
        """Log info message"""
        if 'main' in self.loggers:
            if extra_data:
                message = f"{message} - {json.dumps(extra_data)}"
            self.loggers['main'].info(message)
    
    def log_error(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict] = None):
        """Log error with optional exception details"""
        if 'error' in self.loggers:
            if extra_data:
                message = f"{message} - {json.dumps(extra_data)}"
            if exception:
                self.loggers['error'].error(message, exc_info=exception)
            else:
                self.loggers['error'].error(message)
    
    def log_warning(self, message: str, extra_data: Optional[Dict] = None):
        """Log warning message"""
        if 'main' in self.loggers:
            if extra_data:
                message = f"{message} - {json.dumps(extra_data)}"
            self.loggers['main'].warning(message)
    
    def log_performance(self, operation: str, duration: float, extra_metrics: Optional[Dict] = None):
        """Log performance metrics"""
        if 'performance' in self.loggers:
            metrics = {
                'operation': operation,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            if extra_metrics:
                metrics.update(extra_metrics)
            
            self.loggers['performance'].info(json.dumps(metrics))
    
    def log_safety_event(
        self, 
        severity: str, 
        event_type: str, 
        details: Dict[str, Any],
        intervention_required: bool = False
    ):
        """Log safety-related events"""
        if 'safety' in self.loggers:
            safety_event = {
                'severity': severity,
                'event_type': event_type,
                'intervention_required': intervention_required,
                'timestamp': datetime.now().isoformat(),
                'details': details
            }
            
            log_level = {
                'high': logging.CRITICAL,
                'medium': logging.WARNING,
                'low': logging.INFO
            }.get(severity, logging.INFO)
            
            self.loggers['safety'].log(
                log_level,
                f"Safety Event: {event_type} - {json.dumps(safety_event)}"
            )
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """Save session data to file"""
        try:
            session_file = self.session_directory / f"session_{session_id}.json"
            
            # Prepare session data for saving
            session_export = {
                'session_id': session_id,
                'export_timestamp': datetime.now().isoformat(),
                'session_data': session_data
            }
            
            # Save to file
            with open(session_file, 'w') as f:
                json.dump(session_export, f, indent=2, default=str)
            
            self.log_info(f"Session saved: {session_id}")
            
        except Exception as e:
            self.log_error(f"Failed to save session {session_id}", e)
    
    def log_model_performance(
        self, 
        model_name: str, 
        input_type: str, 
        processing_time: float,
        success: bool,
        confidence: Optional[float] = None
    ):
        """Log individual model performance"""
        performance_data = {
            'model': model_name,
            'input_type': input_type,
            'processing_time': processing_time,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if confidence is not None:
            performance_data['confidence'] = confidence
        
        self.log_performance(f"Model: {model_name}", processing_time, performance_data)
    
    def log_pipeline_execution(
        self,
        pipeline_id: str,
        total_time: float,
        stages_completed: int,
        stages_failed: int,
        parallel_efficiency: float
    ):
        """Log pipeline execution metrics"""
        pipeline_data = {
            'pipeline_id': pipeline_id,
            'total_time': total_time,
            'stages_completed': stages_completed,
            'stages_failed': stages_failed,
            'parallel_efficiency': parallel_efficiency,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_performance("Pipeline Execution", total_time, pipeline_data)
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent logs"""
        try:
            # This is a simplified implementation
            # In production, you might want to use a log analysis library
            
            summary = {
                'period_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'log_files': {
                    'main': str(self.log_directory / 'emotion_sense.log'),
                    'performance': str(self.performance_directory / 'performance.log'),
                    'safety': str(self.log_directory / 'safety.log'),
                    'sessions': str(self.session_directory / 'sessions.log'),
                    'errors': str(self.log_directory / 'errors.log')
                },
                'session_count': len(list(self.session_directory.glob('session_*.json'))),
                'warning': 'Detailed log analysis requires additional implementation'
            }
            
            return summary
            
        except Exception as e:
            self.log_error("Failed to generate log summary", e)
            return {'error': str(e)}