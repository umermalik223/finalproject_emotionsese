import asyncio
import concurrent.futures
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import threading
import queue

logger = logging.getLogger(__name__)

class PipelineManager:
    """High-performance pipeline manager for parallel model execution"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = queue.Queue(maxsize=10)
        self.results_cache = {}
        self.performance_metrics = {}
        
        # Pipeline stages configuration - increased timeouts for reliability
        self.pipeline_stages = {
            'preprocessing': {'timeout': 2.0, 'priority': 1},
            'face_analysis': {'timeout': 5.0, 'priority': 2},
            'speech_analysis': {'timeout': 8.0, 'priority': 2},
            'transcription': {'timeout': 20.0, 'priority': 3},  # Increased for Whisper reliability
            'text_analysis': {'timeout': 3.0, 'priority': 4},
            'mental_analysis': {'timeout': 3.0, 'priority': 4},
            'fusion': {'timeout': 2.0, 'priority': 5},
            'therapeutic_response': {'timeout': 25.0, 'priority': 6}  # Increased for GPT reliability
        }
    
    async def execute_parallel_pipeline(
        self, 
        input_data: Dict[str, Any],
        model_instances: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute multimodal analysis pipeline with parallel processing"""
        
        pipeline_start = time.time()
        logger.info("Starting parallel pipeline execution")
        
        try:
            # Stage 1: Parallel primary analysis (face, speech, transcription)
            primary_tasks = await self._execute_primary_analysis_parallel(
                input_data, model_instances
            )
            
            # Stage 2: Text-dependent analysis (requires transcription)
            secondary_results = await self._execute_secondary_analysis(
                primary_tasks, model_instances
            )
            
            # Stage 3: Fusion and therapeutic response
            final_results = await self._execute_final_stage(
                {**primary_tasks, **secondary_results}, model_instances
            )
            
            # Calculate pipeline performance
            pipeline_time = time.time() - pipeline_start
            final_results['pipeline_metrics'] = {
                'total_time': pipeline_time,
                'stages_completed': len(final_results),
                'parallel_efficiency': self._calculate_efficiency(final_results),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Pipeline completed in {pipeline_time:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return self._get_pipeline_fallback(input_data)
    
    async def _execute_primary_analysis_parallel(
        self, 
        input_data: Dict[str, Any], 
        model_instances: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute face, speech, and transcription analysis in parallel"""
        
        tasks = []
        
        # Face emotion analysis
        if 'video_frame' in input_data:
            tasks.append(
                self._run_with_timeout(
                    'face_analysis',
                    model_instances['face_analyzer'].analyze_emotion,
                    input_data['video_frame']
                )
            )
        
        # Speech emotion analysis
        if 'audio_chunk' in input_data:
            tasks.append(
                self._run_with_timeout(
                    'speech_analysis',
                    model_instances['speech_analyzer'].analyze_audio_chunk,
                    input_data['audio_chunk']
                )
            )
        
        # Speech-to-text transcription
        if 'audio_file' in input_data:
            tasks.append(
                self._run_with_timeout(
                    'transcription',
                    model_instances['stt_analyzer'].transcribe,
                    input_data['audio_file']
                )
            )
        
        # Execute all primary tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        primary_results = {}
        task_index = 0
        
        # Map results based on which tasks were submitted and task order
        task_order = []
        if 'video_frame' in input_data:
            task_order.append('face_emotion')
        if 'audio_chunk' in input_data:
            task_order.append('speech_emotion')
        if 'audio_file' in input_data:
            task_order.append('speech_to_text')
        
        # Map results to their corresponding tasks
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_type = task_order[i] if i < len(task_order) else f"unknown_{i}"
                logger.error(f"Primary task {i} ({task_type}) failed: {result}")
                
                if task_type == 'speech_to_text':
                    primary_results['speech_to_text'] = {
                        'text': '',
                        'confidence': 0.0,
                        'error': f'transcription_failed: {str(result)}'
                    }
                    logger.warning("🔄 Speech-to-text failed, providing fallback")
                elif task_type == 'face_emotion':
                    primary_results['face_emotion'] = {
                        'emotion': 'neutral',
                        'confidence': 0.0,
                        'error': f'face_analysis_failed: {str(result)}'
                    }
                elif task_type == 'speech_emotion':
                    primary_results['speech_emotion'] = {
                        'emotion': 'neutral',
                        'confidence': 0.0,
                        'error': f'speech_analysis_failed: {str(result)}'
                    }
                continue
            elif result and i < len(task_order):
                task_type = task_order[i]
                logger.info(f"Mapping result {i} to {task_type}: {type(result)} = {result}")
                
                if task_type == 'face_emotion':
                    primary_results['face_emotion'] = result
                elif task_type == 'speech_emotion':
                    primary_results['speech_emotion'] = result
                elif task_type == 'speech_to_text':
                    primary_results['speech_to_text'] = result
                    if isinstance(result, dict) and 'text' in result:
                        primary_results['transcribed_text'] = result['text']
                        primary_results['transcription_confidence'] = result.get('confidence', 0.0)
                        logger.info(f"✅ Transcribed text: '{result['text']}'")
                    else:
                        logger.warning(f"Unexpected speech-to-text result format: {result}")
            elif result:
                logger.warning(f"Unmapped result {i}: {result}")
        
        return primary_results
    
    async def _execute_secondary_analysis(
        self, 
        primary_results: Dict[str, Any], 
        model_instances: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute text and mental state analysis (depends on transcription)"""
        
        secondary_results = {}
        transcribed_text = primary_results.get('transcribed_text', '')
        
        if transcribed_text and len(transcribed_text.strip()) > 3:
            # Text emotion analysis
            tasks = [
                self._run_with_timeout(
                    'text_analysis',
                    model_instances['text_analyzer'].analyze_text_emotion,
                    transcribed_text
                )
            ]
            
            # Add mental analysis only if available
            if model_instances.get('mental_analyzer') is not None:
                tasks.append(
                    self._run_with_timeout(
                        'mental_analysis', 
                        model_instances['mental_analyzer'].analyze_mental_state,
                        transcribed_text
                    )
                )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results based on task order
            task_index = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Secondary task {i} failed: {result}")
                elif result:
                    if i == 0:  # First task is always text emotion
                        secondary_results['text_emotion'] = result
                    elif i == 1:  # Second task is mental state (if it exists)
                        secondary_results['mental_state'] = result
                    logger.info(f"Secondary analysis {i}: {result}")
        
        return secondary_results
    
    async def _execute_final_stage(
        self, 
        analysis_results: Dict[str, Any], 
        model_instances: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute emotion fusion and therapeutic response generation"""
        
        try:
            # Emotion fusion
            fusion_result = await self._run_with_timeout(
                'fusion',
                model_instances['emotion_fusion'].fuse_emotions,
                analysis_results
            )
            
            if fusion_result:
                analysis_results['emotion_fusion'] = fusion_result
            
            # Therapeutic response generation (only if available)
            if model_instances.get('therapeutic_generator') is not None:
                therapeutic_result = await self._run_async_with_timeout(
                    'therapeutic_response',
                    model_instances['therapeutic_generator'].generate_therapeutic_response,
                    analysis_results
                )
                
                if therapeutic_result:
                    analysis_results['therapeutic_response'] = therapeutic_result
            
        except Exception as e:
            logger.error(f"Final stage failed: {e}")
        
        return analysis_results
    
    async def _run_with_timeout(
        self, 
        stage_name: str, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Optional[Any]:
        """Run function with timeout and performance tracking"""
        
        stage_config = self.pipeline_stages.get(stage_name, {'timeout': 10.0})
        timeout = stage_config['timeout']
        
        start_time = time.time()
        
        try:
            # Run in thread pool for CPU-bound tasks
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self.thread_pool, func, *args),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            self.performance_metrics[stage_name] = {
                'execution_time': execution_time,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"{stage_name} completed in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"{stage_name} timed out after {timeout}s")
            self.performance_metrics[stage_name] = {
                'execution_time': timeout,
                'success': False,
                'error': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
            return None
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{stage_name} failed: {e}")
            self.performance_metrics[stage_name] = {
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return None
    
    async def _run_async_with_timeout(
        self, 
        stage_name: str, 
        async_func: Callable, 
        *args, 
        **kwargs
    ) -> Optional[Any]:
        """Run async function with timeout and performance tracking"""
        
        stage_config = self.pipeline_stages.get(stage_name, {'timeout': 10.0})
        timeout = stage_config['timeout']
        
        start_time = time.time()
        
        try:
            # Run async function directly with timeout
            result = await asyncio.wait_for(
                async_func(*args, **kwargs),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            self.performance_metrics[stage_name] = {
                'execution_time': execution_time,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"{stage_name} completed in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"{stage_name} timed out after {timeout}s")
            self.performance_metrics[stage_name] = {
                'execution_time': timeout,
                'success': False,
                'error': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
            return None
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{stage_name} failed: {e}")
            self.performance_metrics[stage_name] = {
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return None
    
    def _calculate_efficiency(self, results: Dict[str, Any]) -> float:
        """Calculate pipeline parallel processing efficiency"""
        
        if not self.performance_metrics:
            return 0.0
        
        total_sequential_time = sum(
            metric['execution_time'] 
            for metric in self.performance_metrics.values()
            if metric['success']
        )
        
        actual_pipeline_time = results.get('pipeline_metrics', {}).get('total_time', 1.0)
        
        if actual_pipeline_time > 0:
            efficiency = total_sequential_time / actual_pipeline_time
            return min(efficiency, len(self.performance_metrics))  # Cap at theoretical max
        
        return 0.0
    
    def _get_pipeline_fallback(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback result when entire pipeline fails"""
        fallback_result = {
            'emotion_fusion': {
                'fused_emotion': 'neutral',
                'confidence': 0.3,
                'error': 'pipeline_failed'
            },
            'therapeutic_response': {
                'therapeutic_response': {
                    'empathetic_response': "I'm here to listen and support you.",
                    'calming_techniques': [
                        "Take three deep breaths",
                        "Focus on the present moment",
                        "Remember that you're not alone"
                    ],
                    'emotional_validation': "Your feelings are valid",
                    'severity_assessment': 'low'
                },
                'success': False
            },
            'pipeline_metrics': {
                'total_time': 0.0,
                'stages_completed': 0,
                'parallel_efficiency': 0.0,
                'error': 'complete_pipeline_failure',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add speech_to_text fallback if audio was provided
        if 'audio_file' in input_data or 'audio_chunk' in input_data:
            fallback_result['speech_to_text'] = {
                'text': '',
                'confidence': 0.0,
                'error': 'audio_processing_failed'
            }
            logger.warning("⚠️ Pipeline fallback: Audio provided but processing failed")
        
        return fallback_result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        return {
            'current_metrics': self.performance_metrics,
            'average_times': {
                stage: sum(m['execution_time'] for m in metrics if m['success']) / 
                       max(1, sum(1 for m in metrics if m['success']))
                for stage, metrics in self.performance_metrics.items()
            },
            'success_rates': {
                stage: sum(1 for m in metrics if m['success']) / len(metrics)
                for stage, metrics in self.performance_metrics.items()
            }
        }
    
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)