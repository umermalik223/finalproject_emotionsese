import time
import psutil
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Advanced performance profiler for EmotionSense-AI system"""
    
    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.performance_data = defaultdict(deque)
        self.system_metrics = deque(maxlen=100)  # Last 100 system snapshots
        self.active_timers = {}
        self.session_metrics = {}
        self.lock = threading.Lock()
        
        # Start system monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start background system resource monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._system_monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("System monitoring stopped")
    
    def _system_monitor_loop(self):
        """Background loop for system monitoring"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                system_snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_used_gb': memory.used / (1024**3)
                }
                
                with self.lock:
                    self.system_metrics.append(system_snapshot)
                
                time.sleep(5)  # Sample every 5 seconds
                
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def start_timer(self, operation_name: str) -> str:
        """Start a performance timer"""
        timer_id = f"{operation_name}_{time.time()}"
        self.active_timers[timer_id] = {
            'operation': operation_name,
            'start_time': time.time(),
            'start_datetime': datetime.now().isoformat()
        }
        return timer_id
    
    def end_timer(
        self,
        timer_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """End a performance timer and record results"""
        
        if timer_id not in self.active_timers:
            logger.warning(f"Timer {timer_id} not found")
            return None
        
        timer_data = self.active_timers.pop(timer_id)
        end_time = time.time()
        duration = end_time - timer_data['start_time']
        
        # Create performance record
        performance_record = {
            'operation': timer_data['operation'],
            'duration': duration,
            'success': success,
            'start_time': timer_data['start_datetime'],
            'end_time': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Store record
        with self.lock:
            operation_queue = self.performance_data[timer_data['operation']]
            operation_queue.append(performance_record)
            
            # Maintain max records
            if len(operation_queue) > self.max_records:
                operation_queue.popleft()
        
        return performance_record
    
    def record_processing_time(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record processing time directly"""
        
        performance_record = {
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with self.lock:
            operation_queue = self.performance_data[operation]
            operation_queue.append(performance_record)
            
            if len(operation_queue) > self.max_records:
                operation_queue.popleft()
    
    def profile_function(self, operation_name: str):
        """Decorator for automatic function profiling"""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                timer_id = self.start_timer(operation_name)
                success = True
                result = None
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    success = False
                    raise e
                finally:
                    self.end_timer(timer_id, success=success)
                
                return result
            return wrapper
        return decorator
    
    def get_performance_stats(
        self,
        operation: Optional[str] = None,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get performance statistics for operations"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            if operation:
                operations_to_analyze = [operation] if operation in self.performance_data else []
            else:
                operations_to_analyze = list(self.performance_data.keys())
        
        stats = {}
        
        for op in operations_to_analyze:
            records = self.performance_data[op]
            
            # Filter by time window
            recent_records = [
                r for r in records
                if datetime.fromisoformat(r.get('timestamp', r.get('end_time', ''))) > cutoff_time
            ]
            
            if not recent_records:
                continue
            
            # Calculate statistics
            durations = [r['duration'] for r in recent_records]
            successes = [r['success'] for r in recent_records]
            
            stats[op] = {
                'total_calls': len(recent_records),
                'success_rate': sum(successes) / len(successes) if successes else 0,
                'avg_duration': statistics.mean(durations) if durations else 0,
                'median_duration': statistics.median(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0,
                'percentile_95': statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else (max(durations) if durations else 0)
            }
        
        return stats
    
    def get_system_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get system resource metrics"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_metrics = [
                m for m in self.system_metrics
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
        
        if not recent_metrics:
            return {'error': 'No recent system metrics available'}
        
        # Calculate system statistics
        cpu_values = [m['cpu_percent'] for m in recent_metrics]
        memory_values = [m['memory_percent'] for m in recent_metrics]
        
        return {
            'period_minutes': minutes,
            'samples': len(recent_metrics),
            'cpu': {
                'avg_percent': statistics.mean(cpu_values),
                'max_percent': max(cpu_values),
                'min_percent': min(cpu_values)
            },
            'memory': {
                'avg_percent': statistics.mean(memory_values),
                'max_percent': max(memory_values),
                'min_percent': min(memory_values),
                'current_available_gb': recent_metrics[-1]['memory_available_gb'],
                'current_used_gb': recent_metrics[-1]['memory_used_gb']
            },
            'latest_snapshot': recent_metrics[-1]
        }
    
    def get_session_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get session-specific performance metrics"""
        if session_id and session_id in self.session_metrics:
            return self.session_metrics[session_id]
        elif session_id:
            return {'error': f'Session {session_id} not found'}
        else:
            return {
                'active_sessions': list(self.session_metrics.keys()),
                'total_sessions': len(self.session_metrics)
            }
    
    def start_session_profiling(self, session_id: str):
        """Start profiling for a specific session"""
        self.session_metrics[session_id] = {
            'start_time': datetime.now().isoformat(),
            'operations': defaultdict(list),
            'total_operations': 0,
            'total_duration': 0.0
        }
    
    def record_session_operation(
        self,
        session_id: str,
        operation: str,
        duration: float,
        success: bool = True
    ):
        """Record operation for specific session"""
        if session_id in self.session_metrics:
            session = self.session_metrics[session_id]
            session['operations'][operation].append({
                'duration': duration,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
            session['total_operations'] += 1
            session['total_duration'] += duration
    
    def get_bottleneck_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        stats = self.get_performance_stats(hours=hours)
        
        if not stats:
            return {'error': 'No performance data available'}
        
        # Identify bottlenecks
        bottlenecks = []
        for operation, data in stats.items():
            # High average duration
            if data['avg_duration'] > 2.0:  # > 2 seconds
                bottlenecks.append({
                    'operation': operation,
                    'issue': 'High average duration',
                    'value': data['avg_duration'],
                    'severity': 'high' if data['avg_duration'] > 5.0 else 'medium'
                })
            
            # High variability (high std deviation)
            if data['std_duration'] > data['avg_duration']:
                bottlenecks.append({
                    'operation': operation,
                    'issue': 'High variability',
                    'value': data['std_duration'],
                    'severity': 'medium'
                })
            
            # Low success rate
            if data['success_rate'] < 0.95:
                bottlenecks.append({
                    'operation': operation,
                    'issue': 'Low success rate',
                    'value': data['success_rate'],
                    'severity': 'high' if data['success_rate'] < 0.8 else 'medium'
                })
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['severity']], reverse=True)
        
        return {
            'analysis_period_hours': hours,
            'bottlenecks_found': len(bottlenecks),
            'bottlenecks': bottlenecks,
            'recommendations': self._generate_performance_recommendations(bottlenecks)
        }
    
    def _generate_performance_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            operation = bottleneck['operation']
            issue = bottleneck['issue']
            
            if 'duration' in issue.lower():
                recommendations.append(f"Optimize {operation}: Consider model optimization or hardware upgrade")
            elif 'variability' in issue.lower():
                recommendations.append(f"Stabilize {operation}: Check for resource contention or input quality issues")
            elif 'success rate' in issue.lower():
                recommendations.append(f"Improve {operation}: Add more robust error handling and input validation")
        
        return recommendations
    
    def export_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Export comprehensive performance report"""
        return {
            'report_generated': datetime.now().isoformat(),
            'period_hours': hours,
            'performance_stats': self.get_performance_stats(hours=hours),
            'system_metrics': self.get_system_metrics(minutes=hours*60),
            'bottleneck_analysis': self.get_bottleneck_analysis(hours=hours),
            'active_timers': len(self.active_timers),
            'total_operations_recorded': sum(len(queue) for queue in self.performance_data.values())
        }
    
    def __del__(self):
        """Cleanup profiler"""
        self.stop_system_monitoring()