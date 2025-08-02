"""
Jetson Nano specific utilities and optimizations.
"""

import os
import psutil
import subprocess
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .logger import get_logger

logger = get_logger(__name__)


class JetsonMonitor:
    """Monitor Jetson Nano performance and resources."""
    
    def __init__(self):
        """Initialize Jetson monitor."""
        self.is_jetson = self._detect_jetson()
        self.jetson_stats_available = self._check_jetson_stats()
        
    def _detect_jetson(self) -> bool:
        """Detect if running on Jetson hardware."""
        try:
            # Check for Jetson-specific files
            jetson_files = [
                '/etc/nv_tegra_release',
                '/proc/device-tree/model',
                '/sys/devices/soc0/machine'
            ]
            
            for file_path in jetson_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                        if 'jetson' in content or 'tegra' in content:
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _check_jetson_stats(self) -> bool:
        """Check if jetson-stats is available."""
        try:
            import jtop
            return True
        except ImportError:
            logger.warning("jetson-stats not available. Install with: sudo -H pip install jetson-stats")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'is_jetson': self.is_jetson,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent,
            'python_version': subprocess.check_output(['python3', '--version']).decode().strip()
        }
        
        if self.is_jetson:
            info.update(self._get_jetson_specific_info())
        
        return info
    
    def _get_jetson_specific_info(self) -> Dict[str, Any]:
        """Get Jetson-specific system information."""
        jetson_info = {}
        
        try:
            # Get Jetson model
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    jetson_info['model'] = f.read().strip('\x00')
            
            # Get L4T version
            if os.path.exists('/etc/nv_tegra_release'):
                with open('/etc/nv_tegra_release', 'r') as f:
                    jetson_info['l4t_version'] = f.read().strip()
            
            # Get CUDA version
            try:
                cuda_version = subprocess.check_output(['nvcc', '--version']).decode()
                jetson_info['cuda_version'] = cuda_version.split('release')[1].split(',')[0].strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                jetson_info['cuda_version'] = 'Not available'
            
            # Get GPU information
            try:
                gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits']).decode().strip()
                jetson_info['gpu_info'] = gpu_info
            except (subprocess.CalledProcessError, FileNotFoundError):
                jetson_info['gpu_info'] = 'Not available'
            
        except Exception as e:
            logger.warning(f"Failed to get some Jetson info: {e}")
        
        return jetson_info
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'temperature': self._get_temperature(),
            'power_consumption': self._get_power_consumption()
        }
        
        if self.jetson_stats_available:
            metrics.update(self._get_jetson_stats_metrics())
        
        return metrics
    
    def _get_temperature(self) -> Optional[float]:
        """Get system temperature."""
        try:
            # Try thermal zones
            thermal_zones = Path('/sys/class/thermal')
            if thermal_zones.exists():
                for zone in thermal_zones.glob('thermal_zone*'):
                    temp_file = zone / 'temp'
                    if temp_file.exists():
                        with open(temp_file, 'r') as f:
                            temp = float(f.read().strip()) / 1000.0  # Convert from millidegrees
                            return temp
            
            # Try hwmon
            hwmon_path = Path('/sys/class/hwmon')
            if hwmon_path.exists():
                for hwmon in hwmon_path.glob('hwmon*'):
                    temp_files = list(hwmon.glob('temp*_input'))
                    if temp_files:
                        with open(temp_files[0], 'r') as f:
                            temp = float(f.read().strip()) / 1000.0
                            return temp
            
            return None
            
        except Exception:
            return None
    
    def _get_power_consumption(self) -> Optional[float]:
        """Get power consumption if available."""
        try:
            # Try INA3221 power monitor (common on Jetson)
            ina3221_path = Path('/sys/bus/i2c/drivers/ina3221x')
            if ina3221_path.exists():
                for device in ina3221_path.glob('*/iio:device*'):
                    power_file = device / 'in_power0_input'
                    if power_file.exists():
                        with open(power_file, 'r') as f:
                            power = float(f.read().strip()) / 1000.0  # Convert to watts
                            return power
            
            return None
            
        except Exception:
            return None
    
    def _get_jetson_stats_metrics(self) -> Dict[str, Any]:
        """Get metrics using jetson-stats."""
        try:
            from jtop import jtop
            
            with jtop() as jetson:
                if jetson.ok():
                    return {
                        'gpu_usage': jetson.stats.get('GPU', 0),
                        'emc_usage': jetson.stats.get('EMC', 0),
                        'fan_speed': jetson.fan.get('speed', 0),
                        'power_mode': jetson.nvpmodel.name if hasattr(jetson, 'nvpmodel') else 'Unknown'
                    }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to get jetson-stats metrics: {e}")
            return {}
    
    def optimize_performance(self) -> Dict[str, str]:
        """Apply performance optimizations for Jetson Nano."""
        if not self.is_jetson:
            return {'status': 'Not running on Jetson hardware'}
        
        optimizations = {}
        
        try:
            # Set maximum performance mode
            result = subprocess.run(['sudo', 'nvpmodel', '-m', '0'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                optimizations['nvpmodel'] = 'Set to maximum performance mode (0)'
            else:
                optimizations['nvpmodel'] = f'Failed: {result.stderr}'
            
            # Set maximum CPU frequency
            result = subprocess.run(['sudo', 'jetson_clocks'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                optimizations['jetson_clocks'] = 'CPU clocks maximized'
            else:
                optimizations['jetson_clocks'] = f'Failed: {result.stderr}'
            
            # Disable GUI if running headless
            if os.environ.get('DISPLAY') is None:
                result = subprocess.run(['sudo', 'systemctl', 'set-default', 'multi-user.target'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    optimizations['gui'] = 'GUI disabled for headless operation'
            
            logger.info("Performance optimizations applied")
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            optimizations['error'] = str(e)
        
        return optimizations
    
    def monitor_inference(self, duration: float = 60.0, interval: float = 1.0) -> List[Dict[str, Any]]:
        """
        Monitor system performance during inference.
        
        Args:
            duration: Monitoring duration in seconds
            interval: Sampling interval in seconds
            
        Returns:
            List of performance samples
        """
        logger.info(f"Starting performance monitoring for {duration} seconds")
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            sample = self.get_performance_metrics()
            samples.append(sample)
            time.sleep(interval)
        
        logger.info(f"Collected {len(samples)} performance samples")
        return samples
    
    def save_performance_report(self, samples: List[Dict[str, Any]], output_path: str):
        """
        Save performance monitoring report.
        
        Args:
            samples: Performance samples
            output_path: Output file path
        """
        report = {
            'system_info': self.get_system_info(),
            'monitoring_duration': len(samples),
            'samples': samples,
            'summary': self._calculate_performance_summary(samples)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to: {output_path}")
    
    def _calculate_performance_summary(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance summary statistics."""
        if not samples:
            return {}
        
        import statistics
        
        cpu_values = [s['cpu_percent'] for s in samples if 'cpu_percent' in s]
        memory_values = [s['memory_percent'] for s in samples if 'memory_percent' in s]
        temp_values = [s['temperature'] for s in samples if s.get('temperature') is not None]
        
        summary = {}
        
        if cpu_values:
            summary['cpu'] = {
                'mean': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            }
        
        if memory_values:
            summary['memory'] = {
                'mean': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            }
        
        if temp_values:
            summary['temperature'] = {
                'mean': statistics.mean(temp_values),
                'max': max(temp_values),
                'min': min(temp_values),
                'std': statistics.stdev(temp_values) if len(temp_values) > 1 else 0
            }
        
        return summary


class JetsonOptimizer:
    """Jetson-specific optimization utilities."""
    
    @staticmethod
    def setup_swap(swap_size_gb: int = 4):
        """Set up swap file for additional memory."""
        try:
            swap_file = '/swapfile'
            
            # Check if swap already exists
            result = subprocess.run(['swapon', '--show'], capture_output=True, text=True)
            if swap_file in result.stdout:
                logger.info("Swap file already exists")
                return
            
            # Create swap file
            logger.info(f"Creating {swap_size_gb}GB swap file...")
            subprocess.run(['sudo', 'fallocate', '-l', f'{swap_size_gb}G', swap_file], check=True)
            subprocess.run(['sudo', 'chmod', '600', swap_file], check=True)
            subprocess.run(['sudo', 'mkswap', swap_file], check=True)
            subprocess.run(['sudo', 'swapon', swap_file], check=True)
            
            # Make permanent
            with open('/etc/fstab', 'a') as f:
                f.write(f'{swap_file} none swap sw 0 0\n')
            
            logger.info("Swap file created successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup swap: {e}")
        except Exception as e:
            logger.error(f"Unexpected error setting up swap: {e}")
    
    @staticmethod
    def optimize_opencv():
        """Optimize OpenCV for Jetson Nano."""
        try:
            import cv2
            
            # Set number of threads
            cv2.setNumThreads(4)  # Jetson Nano has 4 CPU cores
            
            # Use optimized code paths
            cv2.setUseOptimized(True)
            
            logger.info("OpenCV optimizations applied")
            
        except ImportError:
            logger.warning("OpenCV not available for optimization")
        except Exception as e:
            logger.error(f"Failed to optimize OpenCV: {e}")
    
    @staticmethod
    def setup_environment():
        """Set up optimal environment variables for Jetson Nano."""
        env_vars = {
            'CUDA_CACHE_DISABLE': '0',
            'CUDA_CACHE_MAXSIZE': '2147483648',  # 2GB
            'OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_USE_EXTERNAL_CONTEXT': '1',
            'OMP_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            logger.info(f"Set {var}={value}")


# Global instances
jetson_monitor = JetsonMonitor()
jetson_optimizer = JetsonOptimizer()
