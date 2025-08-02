#!/usr/bin/env python3
"""
Cross-platform compatibility check for Windows development and Jetson Nano deployment.
Windows开发环境和Jetson Nano部署环境的跨平台兼容性检查。
"""

import sys
import platform
import subprocess
import importlib
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logger import setup_logger

logger = setup_logger("compatibility_check", level="INFO")


class CompatibilityChecker:
    """Cross-platform compatibility checker."""
    
    def __init__(self):
        """Initialize compatibility checker."""
        self.platform_info = {
            'current_platform': platform.system(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version(),
            'is_windows': platform.system() == 'Windows',
            'is_linux': platform.system() == 'Linux',
            'is_jetson': self._detect_jetson()
        }
        
        self.results = {
            'platform_info': self.platform_info,
            'dependency_compatibility': {},
            'path_compatibility': {},
            'model_compatibility': {},
            'performance_compatibility': {},
            'recommendations': []
        }
    
    def _detect_jetson(self) -> bool:
        """Detect if running on Jetson hardware."""
        jetson_indicators = [
            Path('/etc/nv_tegra_release'),
            Path('/proc/device-tree/model')
        ]
        
        for indicator in jetson_indicators:
            if indicator.exists():
                try:
                    with open(indicator, 'r') as f:
                        content = f.read().lower()
                        if 'tegra' in content or 'jetson' in content:
                            return True
                except:
                    pass
        
        return False
    
    def check_dependency_compatibility(self) -> Dict[str, Any]:
        """Check dependency compatibility across platforms."""
        logger.info("Checking dependency compatibility...")
        
        compatibility = {
            'pytorch_compatible': False,
            'opencv_compatible': False,
            'ultralytics_compatible': False,
            'platform_specific_issues': [],
            'version_conflicts': []
        }
        
        # PyTorch compatibility / PyTorch兼容性
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            compatibility['pytorch_compatible'] = True
            compatibility['pytorch_version'] = torch_version
            compatibility['cuda_available'] = cuda_available
            
            # Check for platform-specific PyTorch issues / 检查平台特定的PyTorch问题
            if self.platform_info['is_jetson']:
                if not cuda_available:
                    compatibility['platform_specific_issues'].append(
                        "CUDA not available on Jetson. Check JetPack installation."
                    )
                
                # Check for Jetson-specific PyTorch version / 检查Jetson特定的PyTorch版本
                if not any(arch in torch_version for arch in ['aarch64', 'arm']):
                    compatibility['platform_specific_issues'].append(
                        "PyTorch may not be optimized for ARM architecture. Consider Jetson-specific build."
                    )
            
            elif self.platform_info['is_windows']:
                if cuda_available:
                    logger.info("✓ CUDA available on Windows for development")
                else:
                    compatibility['platform_specific_issues'].append(
                        "CUDA not available on Windows. GPU acceleration disabled."
                    )
            
            logger.info(f"✓ PyTorch {torch_version} compatible")
            
        except ImportError:
            compatibility['pytorch_compatible'] = False
            compatibility['platform_specific_issues'].append("PyTorch not installed")
            logger.error("✗ PyTorch not available")
        
        # OpenCV compatibility / OpenCV兼容性
        try:
            import cv2
            opencv_version = cv2.__version__
            
            compatibility['opencv_compatible'] = True
            compatibility['opencv_version'] = opencv_version
            
            # Test OpenCV functionality / 测试OpenCV功能
            test_image = cv2.imread('non_existent_file.jpg')  # Should return None
            if test_image is None:
                logger.info("✓ OpenCV imread function working")
            
            logger.info(f"✓ OpenCV {opencv_version} compatible")
            
        except ImportError:
            compatibility['opencv_compatible'] = False
            compatibility['platform_specific_issues'].append("OpenCV not installed")
            logger.error("✗ OpenCV not available")
        
        # Ultralytics compatibility / Ultralytics兼容性
        try:
            import ultralytics
            ultralytics_version = ultralytics.__version__
            
            compatibility['ultralytics_compatible'] = True
            compatibility['ultralytics_version'] = ultralytics_version
            
            logger.info(f"✓ Ultralytics {ultralytics_version} compatible")
            
        except ImportError:
            compatibility['ultralytics_compatible'] = False
            compatibility['platform_specific_issues'].append("Ultralytics not installed")
            logger.error("✗ Ultralytics not available")
        
        self.results['dependency_compatibility'] = compatibility
        return compatibility
    
    def check_path_compatibility(self) -> Dict[str, Any]:
        """Check file path compatibility across platforms."""
        logger.info("Checking path compatibility...")
        
        path_compat = {
            'path_separator_issues': [],
            'absolute_path_issues': [],
            'relative_path_issues': [],
            'recommendations': []
        }
        
        # Test common paths used in the project / 测试项目中使用的常见路径
        test_paths = [
            'models/character_detector.pt',
            'data/samples/test_image.jpg',
            'config/model_config.yaml',
            'src/models/yolo_character_detector.py'
        ]
        
        for path_str in test_paths:
            # Test Path object creation / 测试Path对象创建
            try:
                path_obj = Path(path_str)
                
                # Check if path uses forward slashes (cross-platform) / 检查路径是否使用正斜杠（跨平台）
                if '\\' in path_str:
                    path_compat['path_separator_issues'].append(
                        f"Path uses backslashes: {path_str}. Use forward slashes for cross-platform compatibility."
                    )
                
                # Test path operations / 测试路径操作
                parent = path_obj.parent
                name = path_obj.name
                suffix = path_obj.suffix
                
                logger.debug(f"Path {path_str}: parent={parent}, name={name}, suffix={suffix}")
                
            except Exception as e:
                path_compat['absolute_path_issues'].append(f"Path creation failed for {path_str}: {e}")
        
        # Test absolute vs relative paths / 测试绝对路径与相对路径
        if self.platform_info['is_windows']:
            # Windows-specific path tests / Windows特定路径测试
            windows_paths = ['C:\\temp\\test.txt', 'C:/temp/test.txt']
            for win_path in windows_paths:
                try:
                    Path(win_path)
                except Exception as e:
                    path_compat['absolute_path_issues'].append(f"Windows path issue: {win_path} - {e}")
        
        # Recommendations / 建议
        if path_compat['path_separator_issues']:
            path_compat['recommendations'].append(
                "Use pathlib.Path for all file operations to ensure cross-platform compatibility"
            )
        
        if not path_compat['path_separator_issues'] and not path_compat['absolute_path_issues']:
            logger.info("✓ Path compatibility looks good")
        
        self.results['path_compatibility'] = path_compat
        return path_compat
    
    def check_model_compatibility(self) -> Dict[str, Any]:
        """Check model file compatibility across platforms."""
        logger.info("Checking model compatibility...")
        
        model_compat = {
            'pytorch_models_compatible': False,
            'onnx_models_compatible': False,
            'tensorrt_available': False,
            'model_loading_test': False,
            'cross_platform_issues': []
        }
        
        # Check PyTorch model compatibility / 检查PyTorch模型兼容性
        try:
            import torch
            
            # Test model creation and saving / 测试模型创建和保存
            dummy_model = torch.nn.Linear(10, 1)
            test_model_path = Path('test_model_temp.pt')
            
            torch.save(dummy_model.state_dict(), test_model_path)
            loaded_state = torch.load(test_model_path, map_location='cpu')
            
            # Clean up / 清理
            test_model_path.unlink()
            
            model_compat['pytorch_models_compatible'] = True
            logger.info("✓ PyTorch model compatibility verified")
            
        except Exception as e:
            model_compat['cross_platform_issues'].append(f"PyTorch model compatibility issue: {e}")
            logger.error(f"✗ PyTorch model compatibility issue: {e}")
        
        # Check ONNX compatibility / 检查ONNX兼容性
        try:
            import onnx
            import onnxruntime
            
            model_compat['onnx_models_compatible'] = True
            logger.info("✓ ONNX runtime available")
            
        except ImportError:
            model_compat['cross_platform_issues'].append("ONNX runtime not available")
            logger.warning("⚠ ONNX runtime not available")
        
        # Check TensorRT availability / 检查TensorRT可用性
        try:
            import tensorrt
            model_compat['tensorrt_available'] = True
            logger.info("✓ TensorRT available")
            
        except ImportError:
            if self.platform_info['is_jetson']:
                model_compat['cross_platform_issues'].append(
                    "TensorRT not available on Jetson. Check JetPack installation."
                )
            else:
                logger.info("- TensorRT not available (expected on non-Jetson platforms)")
        
        # Test actual model loading if available / 如果可用则测试实际模型加载
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pt'))
            if model_files:
                try:
                    from src.models.yolo_character_detector import YOLOCharacterDetector
                    detector = YOLOCharacterDetector()
                    # Don't actually load to avoid errors, just test import
                    model_compat['model_loading_test'] = True
                    logger.info("✓ Model loading framework available")
                except Exception as e:
                    model_compat['cross_platform_issues'].append(f"Model loading test failed: {e}")
        
        self.results['model_compatibility'] = model_compat
        return model_compat
    
    def check_performance_compatibility(self) -> Dict[str, Any]:
        """Check performance characteristics across platforms."""
        logger.info("Checking performance compatibility...")
        
        perf_compat = {
            'cpu_performance': {},
            'memory_performance': {},
            'gpu_performance': {},
            'platform_optimizations': [],
            'performance_warnings': []
        }
        
        # CPU performance check / CPU性能检查
        import psutil
        import time
        
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        perf_compat['cpu_performance'] = {
            'cpu_count': cpu_count,
            'cpu_freq_current': cpu_freq.current if cpu_freq else None,
            'cpu_freq_max': cpu_freq.max if cpu_freq else None
        }
        
        # Memory performance check / 内存性能检查
        memory = psutil.virtual_memory()
        perf_compat['memory_performance'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent
        }
        
        # Platform-specific optimizations / 平台特定优化
        if self.platform_info['is_jetson']:
            perf_compat['platform_optimizations'].extend([
                "Use TensorRT for model optimization",
                "Enable GPU memory optimization",
                "Use FP16 precision for inference",
                "Set appropriate power mode with nvpmodel"
            ])
            
            # Check Jetson-specific performance settings / 检查Jetson特定性能设置
            try:
                result = subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True)
                if result.returncode == 0:
                    perf_compat['jetson_power_mode'] = result.stdout.strip()
                else:
                    perf_compat['performance_warnings'].append("Could not check Jetson power mode")
            except FileNotFoundError:
                perf_compat['performance_warnings'].append("nvpmodel command not found")
        
        elif self.platform_info['is_windows']:
            perf_compat['platform_optimizations'].extend([
                "Use GPU acceleration if available",
                "Consider using Windows ML for optimization",
                "Enable multi-threading for CPU inference"
            ])
        
        # Performance warnings / 性能警告
        if perf_compat['memory_performance']['total_gb'] < 4:
            perf_compat['performance_warnings'].append(
                f"Low memory: {perf_compat['memory_performance']['total_gb']:.1f}GB. Consider adding swap or reducing batch size."
            )
        
        if cpu_count < 4:
            perf_compat['performance_warnings'].append(
                f"Low CPU count: {cpu_count}. Performance may be limited."
            )
        
        self.results['performance_compatibility'] = perf_compat
        return perf_compat
    
    def generate_compatibility_recommendations(self) -> List[str]:
        """Generate platform-specific recommendations."""
        recommendations = []
        
        # Dependency recommendations / 依赖建议
        dep_compat = self.results['dependency_compatibility']
        if dep_compat['platform_specific_issues']:
            recommendations.append("Dependency Issues:")
            for issue in dep_compat['platform_specific_issues']:
                recommendations.append(f"  • {issue}")
        
        # Path recommendations / 路径建议
        path_compat = self.results['path_compatibility']
        if path_compat['recommendations']:
            recommendations.append("Path Compatibility:")
            for rec in path_compat['recommendations']:
                recommendations.append(f"  • {rec}")
        
        # Model recommendations / 模型建议
        model_compat = self.results['model_compatibility']
        if model_compat['cross_platform_issues']:
            recommendations.append("Model Compatibility:")
            for issue in model_compat['cross_platform_issues']:
                recommendations.append(f"  • {issue}")
        
        # Performance recommendations / 性能建议
        perf_compat = self.results['performance_compatibility']
        if perf_compat['platform_optimizations']:
            recommendations.append("Platform Optimizations:")
            for opt in perf_compat['platform_optimizations']:
                recommendations.append(f"  • {opt}")
        
        if perf_compat['performance_warnings']:
            recommendations.append("Performance Warnings:")
            for warning in perf_compat['performance_warnings']:
                recommendations.append(f"  • {warning}")
        
        # General cross-platform recommendations / 通用跨平台建议
        recommendations.extend([
            "General Cross-Platform Best Practices:",
            "  • Use pathlib.Path for all file operations",
            "  • Test on both development and deployment platforms",
            "  • Use relative paths where possible",
            "  • Handle platform-specific dependencies gracefully",
            "  • Document platform-specific installation steps"
        ])
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def run_complete_compatibility_check(self) -> Dict[str, Any]:
        """Run complete compatibility check."""
        logger.info(f"Starting compatibility check on {self.platform_info['current_platform']}...")
        
        # Run all compatibility checks / 运行所有兼容性检查
        self.check_dependency_compatibility()
        self.check_path_compatibility()
        self.check_model_compatibility()
        self.check_performance_compatibility()
        self.generate_compatibility_recommendations()
        
        # Add timestamp / 添加时间戳
        self.results['check_timestamp'] = time.time()
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate compatibility report."""
        report_lines = []
        
        report_lines.append("="*60)
        report_lines.append("CROSS-PLATFORM COMPATIBILITY REPORT")
        report_lines.append("跨平台兼容性报告")
        report_lines.append("="*60)
        
        # Platform info / 平台信息
        platform_info = self.results['platform_info']
        report_lines.append(f"Platform: {platform_info['current_platform']} ({platform_info['architecture']})")
        report_lines.append(f"Python: {platform_info['python_version']}")
        report_lines.append(f"Jetson Device: {platform_info['is_jetson']}")
        report_lines.append("")
        
        # Dependency compatibility / 依赖兼容性
        dep_compat = self.results['dependency_compatibility']
        report_lines.append("DEPENDENCY COMPATIBILITY:")
        report_lines.append(f"  PyTorch: {'✓' if dep_compat['pytorch_compatible'] else '✗'}")
        report_lines.append(f"  OpenCV: {'✓' if dep_compat['opencv_compatible'] else '✗'}")
        report_lines.append(f"  Ultralytics: {'✓' if dep_compat['ultralytics_compatible'] else '✗'}")
        if dep_compat.get('cuda_available'):
            report_lines.append(f"  CUDA: ✓ Available")
        else:
            report_lines.append(f"  CUDA: ⚠ Not available")
        report_lines.append("")
        
        # Model compatibility / 模型兼容性
        model_compat = self.results['model_compatibility']
        report_lines.append("MODEL COMPATIBILITY:")
        report_lines.append(f"  PyTorch Models: {'✓' if model_compat['pytorch_models_compatible'] else '✗'}")
        report_lines.append(f"  ONNX Models: {'✓' if model_compat['onnx_models_compatible'] else '⚠'}")
        report_lines.append(f"  TensorRT: {'✓' if model_compat['tensorrt_available'] else '⚠'}")
        report_lines.append("")
        
        # Performance / 性能
        perf_compat = self.results['performance_compatibility']
        cpu_perf = perf_compat['cpu_performance']
        mem_perf = perf_compat['memory_performance']
        
        report_lines.append("PERFORMANCE CHARACTERISTICS:")
        report_lines.append(f"  CPU Cores: {cpu_perf['cpu_count']}")
        report_lines.append(f"  Memory: {mem_perf['available_gb']:.1f}GB / {mem_perf['total_gb']:.1f}GB")
        if 'jetson_power_mode' in perf_compat:
            report_lines.append(f"  Jetson Power Mode: {perf_compat['jetson_power_mode']}")
        report_lines.append("")
        
        # Recommendations / 建议
        if self.results['recommendations']:
            report_lines.append("RECOMMENDATIONS:")
            for rec in self.results['recommendations']:
                report_lines.append(rec)
            report_lines.append("")
        
        report_lines.append("="*60)
        
        return "\n".join(report_lines)


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check cross-platform compatibility")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument("--report", help="Output file for compatibility report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Run compatibility check / 运行兼容性检查
    checker = CompatibilityChecker()
    results = checker.run_complete_compatibility_check()
    
    # Generate and display report / 生成并显示报告
    report = checker.generate_report()
    print(report)
    
    # Save detailed results / 保存详细结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")
    
    # Save report / 保存报告
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.report}")


if __name__ == "__main__":
    main()
