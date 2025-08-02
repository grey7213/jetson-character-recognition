"""
Performance monitoring and optimization utilities.
"""

import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    inference_time: float
    preprocessing_time: float
    postprocessing_time: float
    total_time: float
    fps: float
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    temperature: Optional[float] = None


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        
        # Timing contexts
        self._timers = {}
        
    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing operations.
        
        Args:
            name: Timer name
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self._timers[name] = end_time - start_time
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self._timers[f"{name}_start"] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
        """
        start_key = f"{name}_start"
        if start_key not in self._timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self._timers[start_key]
        self._timers[name] = elapsed
        del self._timers[start_key]
        return elapsed
    
    def get_timer(self, name: str) -> float:
        """Get the value of a timer."""
        return self._timers.get(name, 0.0)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """
        Record performance metrics.
        
        Args:
            metrics: Performance metrics to record
        """
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Maintain max history size
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the latest recorded metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get average metrics over the last N samples.
        
        Args:
            last_n: Number of recent samples to average (None for all)
            
        Returns:
            Dictionary of averaged metrics
        """
        with self._lock:
            if not self.metrics_history:
                return {}
            
            samples = self.metrics_history[-last_n:] if last_n else self.metrics_history
            
            if not samples:
                return {}
            
            # Calculate averages
            avg_metrics = {
                'inference_time': np.mean([m.inference_time for m in samples]),
                'preprocessing_time': np.mean([m.preprocessing_time for m in samples]),
                'postprocessing_time': np.mean([m.postprocessing_time for m in samples]),
                'total_time': np.mean([m.total_time for m in samples]),
                'fps': np.mean([m.fps for m in samples])
            }
            
            # Add optional metrics if available
            memory_values = [m.memory_usage for m in samples if m.memory_usage is not None]
            if memory_values:
                avg_metrics['memory_usage'] = np.mean(memory_values)
            
            gpu_values = [m.gpu_usage for m in samples if m.gpu_usage is not None]
            if gpu_values:
                avg_metrics['gpu_usage'] = np.mean(gpu_values)
            
            temp_values = [m.temperature for m in samples if m.temperature is not None]
            if temp_values:
                avg_metrics['temperature'] = np.mean(temp_values)
            
            return avg_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            if not self.metrics_history:
                return {'status': 'No metrics recorded'}
            
            # Recent performance (last 100 samples)
            recent_avg = self.get_average_metrics(last_n=100)
            
            # Overall performance
            overall_avg = self.get_average_metrics()
            
            # Performance trends
            fps_values = [m.fps for m in self.metrics_history]
            inference_times = [m.inference_time for m in self.metrics_history]
            
            summary = {
                'total_samples': len(self.metrics_history),
                'recent_performance': recent_avg,
                'overall_performance': overall_avg,
                'performance_trends': {
                    'fps_trend': self._calculate_trend(fps_values),
                    'inference_time_trend': self._calculate_trend(inference_times),
                    'fps_std': float(np.std(fps_values)),
                    'inference_time_std': float(np.std(inference_times))
                }
            }
            
            return summary
    
    def _calculate_trend(self, values: List[float], window_size: int = 50) -> str:
        """
        Calculate performance trend.
        
        Args:
            values: List of values
            window_size: Window size for trend calculation
            
        Returns:
            Trend description
        """
        if len(values) < window_size * 2:
            return "insufficient_data"
        
        # Compare recent window with earlier window
        recent_avg = np.mean(values[-window_size:])
        earlier_avg = np.mean(values[-window_size*2:-window_size])
        
        change_percent = ((recent_avg - earlier_avg) / earlier_avg) * 100
        
        if abs(change_percent) < 2:
            return "stable"
        elif change_percent > 0:
            return f"improving_{change_percent:.1f}%"
        else:
            return f"degrading_{abs(change_percent):.1f}%"
    
    def reset_metrics(self):
        """Reset all recorded metrics."""
        with self._lock:
            self.metrics_history.clear()
            self._timers.clear()
        logger.info("Performance metrics reset")


class FPSCounter:
    """Simple FPS counter for real-time monitoring."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        Update FPS counter and return current FPS.
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        
        # Maintain window size
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return 0.0
    
    def get_fps(self) -> float:
        """Get current FPS without updating."""
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return 0.0


class PerformanceOptimizer:
    """Automatic performance optimization utilities."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize performance optimizer.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.optimization_history = []
        
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """
        Suggest performance optimizations based on current metrics.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        summary = self.monitor.get_performance_summary()
        
        if 'recent_performance' not in summary:
            return suggestions
        
        recent = summary['recent_performance']
        
        # FPS-based suggestions
        if 'fps' in recent:
            fps = recent['fps']
            if fps < 5:
                suggestions.append({
                    'type': 'critical',
                    'issue': 'Very low FPS',
                    'suggestion': 'Consider reducing input resolution or using a smaller model',
                    'priority': 'high'
                })
            elif fps < 10:
                suggestions.append({
                    'type': 'warning',
                    'issue': 'Low FPS',
                    'suggestion': 'Enable TensorRT optimization or reduce batch size',
                    'priority': 'medium'
                })
        
        # Memory-based suggestions
        if 'memory_usage' in recent:
            memory = recent['memory_usage']
            if memory > 90:
                suggestions.append({
                    'type': 'critical',
                    'issue': 'High memory usage',
                    'suggestion': 'Reduce batch size or enable memory optimization',
                    'priority': 'high'
                })
            elif memory > 75:
                suggestions.append({
                    'type': 'warning',
                    'issue': 'Moderate memory usage',
                    'suggestion': 'Monitor memory usage and consider optimization',
                    'priority': 'low'
                })
        
        # Temperature-based suggestions
        if 'temperature' in recent:
            temp = recent['temperature']
            if temp > 80:
                suggestions.append({
                    'type': 'critical',
                    'issue': 'High temperature',
                    'suggestion': 'Improve cooling or reduce processing load',
                    'priority': 'high'
                })
            elif temp > 70:
                suggestions.append({
                    'type': 'warning',
                    'issue': 'Elevated temperature',
                    'suggestion': 'Monitor temperature and ensure adequate cooling',
                    'priority': 'medium'
                })
        
        # Inference time suggestions
        if 'inference_time' in recent:
            inference_time = recent['inference_time']
            if inference_time > 0.2:  # 200ms
                suggestions.append({
                    'type': 'warning',
                    'issue': 'Slow inference',
                    'suggestion': 'Enable GPU acceleration or model optimization',
                    'priority': 'medium'
                })
        
        return suggestions
    
    def auto_optimize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically apply optimizations based on performance.
        
        Args:
            config: Current configuration
            
        Returns:
            Updated configuration with optimizations
        """
        suggestions = self.suggest_optimizations()
        optimized_config = config.copy()
        applied_optimizations = []
        
        for suggestion in suggestions:
            if suggestion['priority'] == 'high':
                # Apply high-priority optimizations automatically
                if 'reduce batch size' in suggestion['suggestion'].lower():
                    current_batch = optimized_config.get('batch_size', 1)
                    if current_batch > 1:
                        optimized_config['batch_size'] = max(1, current_batch // 2)
                        applied_optimizations.append('Reduced batch size')
                
                if 'reduce input resolution' in suggestion['suggestion'].lower():
                    current_size = optimized_config.get('input_size', [640, 640])
                    if current_size[0] > 320:
                        new_size = [max(320, current_size[0] // 2), max(320, current_size[1] // 2)]
                        optimized_config['input_size'] = new_size
                        applied_optimizations.append('Reduced input resolution')
                
                if 'tensorrt' in suggestion['suggestion'].lower():
                    optimized_config['use_tensorrt'] = True
                    applied_optimizations.append('Enabled TensorRT')
        
        return {
            'config': optimized_config,
            'applied_optimizations': applied_optimizations,
            'suggestions': suggestions
        }
