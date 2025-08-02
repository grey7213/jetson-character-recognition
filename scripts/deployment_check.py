#!/usr/bin/env python3
"""
Comprehensive deployment readiness check for Jetson Character Recognition system.
Jetson字符识别系统的综合部署就绪性检查。
"""

import sys
import subprocess
import importlib
import platform
import psutil
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logger import setup_logger

logger = setup_logger("deployment_check", level="INFO")


class DeploymentChecker:
    """Comprehensive deployment readiness checker."""
    
    def __init__(self):
        """Initialize deployment checker."""
        self.results = {
            'system_info': {},
            'dependencies': {},
            'hardware': {},
            'models': {},
            'data': {},
            'configuration': {},
            'performance': {},
            'overall_status': 'UNKNOWN'
        }
        
        self.required_packages = [
            'torch', 'torchvision', 'ultralytics', 'opencv-python', 
            'numpy', 'pillow', 'pyyaml', 'psutil'
        ]
        
        self.optional_packages = [
            'tensorrt', 'pycuda', 'onnx', 'onnxruntime'
        ]
    
    def check_system_info(self) -> Dict[str, Any]:
        """Check system information."""
        logger.info("Checking system information...")
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2)
        }
        
        # Check if running on Jetson / 检查是否在Jetson上运行
        jetson_release_file = Path('/etc/nv_tegra_release')
        if jetson_release_file.exists():
            try:
                with open(jetson_release_file, 'r') as f:
                    jetson_info = f.read().strip()
                system_info['jetson_info'] = jetson_info
                system_info['is_jetson'] = True
            except:
                system_info['is_jetson'] = False
        else:
            system_info['is_jetson'] = False
        
        # Check CUDA availability / 检查CUDA可用性
        try:
            import torch
            system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info['cuda_device_count'] = torch.cuda.device_count()
                system_info['cuda_device_name'] = torch.cuda.get_device_name(0)
                system_info['cuda_version'] = torch.version.cuda
        except ImportError:
            system_info['cuda_available'] = False
        
        self.results['system_info'] = system_info
        
        # Log system info / 记录系统信息
        logger.info(f"Platform: {system_info['platform']}")
        logger.info(f"Python: {system_info['python_version']}")
        logger.info(f"Memory: {system_info['memory_available_gb']:.1f}GB available / {system_info['memory_total_gb']:.1f}GB total")
        logger.info(f"Jetson device: {system_info['is_jetson']}")
        logger.info(f"CUDA available: {system_info.get('cuda_available', False)}")
        
        return system_info
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check required and optional dependencies."""
        logger.info("Checking dependencies...")
        
        dependencies = {
            'required': {},
            'optional': {},
            'missing_required': [],
            'missing_optional': []
        }
        
        # Check required packages / 检查必需包
        for package in self.required_packages:
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                dependencies['required'][package] = {
                    'installed': True,
                    'version': version
                }
                logger.info(f"✓ {package}: {version}")
            except ImportError:
                dependencies['required'][package] = {
                    'installed': False,
                    'version': None
                }
                dependencies['missing_required'].append(package)
                logger.warning(f"✗ {package}: NOT INSTALLED")
        
        # Check optional packages / 检查可选包
        for package in self.optional_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                dependencies['optional'][package] = {
                    'installed': True,
                    'version': version
                }
                logger.info(f"✓ {package} (optional): {version}")
            except ImportError:
                dependencies['optional'][package] = {
                    'installed': False,
                    'version': None
                }
                dependencies['missing_optional'].append(package)
                logger.info(f"- {package} (optional): not installed")
        
        self.results['dependencies'] = dependencies
        return dependencies
    
    def check_hardware_requirements(self) -> Dict[str, Any]:
        """Check hardware requirements."""
        logger.info("Checking hardware requirements...")
        
        hardware = {
            'memory_sufficient': False,
            'disk_space_sufficient': False,
            'cpu_sufficient': False,
            'gpu_available': False,
            'recommendations': []
        }
        
        system_info = self.results['system_info']
        
        # Memory check / 内存检查
        min_memory_gb = 2.0
        recommended_memory_gb = 4.0
        available_memory = system_info['memory_total_gb']
        
        if available_memory >= recommended_memory_gb:
            hardware['memory_sufficient'] = True
            logger.info(f"✓ Memory: {available_memory:.1f}GB (recommended: {recommended_memory_gb}GB)")
        elif available_memory >= min_memory_gb:
            hardware['memory_sufficient'] = True
            hardware['recommendations'].append(f"Consider upgrading to {recommended_memory_gb}GB RAM for better performance")
            logger.warning(f"⚠ Memory: {available_memory:.1f}GB (minimum: {min_memory_gb}GB)")
        else:
            hardware['memory_sufficient'] = False
            hardware['recommendations'].append(f"Insufficient memory. Need at least {min_memory_gb}GB")
            logger.error(f"✗ Memory: {available_memory:.1f}GB (minimum: {min_memory_gb}GB)")
        
        # Disk space check / 磁盘空间检查
        min_disk_gb = 5.0
        available_disk = system_info['disk_free_gb']
        
        if available_disk >= min_disk_gb:
            hardware['disk_space_sufficient'] = True
            logger.info(f"✓ Disk space: {available_disk:.1f}GB available")
        else:
            hardware['disk_space_sufficient'] = False
            hardware['recommendations'].append(f"Insufficient disk space. Need at least {min_disk_gb}GB")
            logger.error(f"✗ Disk space: {available_disk:.1f}GB (minimum: {min_disk_gb}GB)")
        
        # CPU check / CPU检查
        min_cpu_cores = 2
        cpu_count = system_info['cpu_count']
        
        if cpu_count >= min_cpu_cores:
            hardware['cpu_sufficient'] = True
            logger.info(f"✓ CPU cores: {cpu_count}")
        else:
            hardware['cpu_sufficient'] = False
            hardware['recommendations'].append(f"Need at least {min_cpu_cores} CPU cores")
            logger.error(f"✗ CPU cores: {cpu_count} (minimum: {min_cpu_cores})")
        
        # GPU check / GPU检查
        if system_info.get('cuda_available', False):
            hardware['gpu_available'] = True
            logger.info(f"✓ GPU: {system_info.get('cuda_device_name', 'Available')}")
        else:
            hardware['gpu_available'] = False
            hardware['recommendations'].append("GPU acceleration not available. Performance will be limited.")
            logger.warning("⚠ GPU: Not available or CUDA not installed")
        
        self.results['hardware'] = hardware
        return hardware
    
    def check_models(self) -> Dict[str, Any]:
        """Check model files availability."""
        logger.info("Checking model files...")
        
        models = {
            'models_directory_exists': False,
            'available_models': [],
            'missing_models': [],
            'recommendations': []
        }
        
        models_dir = Path('models')
        if models_dir.exists():
            models['models_directory_exists'] = True
            
            # Check for common model files / 检查常见模型文件
            expected_models = [
                'yolov8n_character.pt',
                'character_detector.pt',
                'character_detector_best.pt'
            ]
            
            for model_file in expected_models:
                model_path = models_dir / model_file
                if model_path.exists():
                    models['available_models'].append(model_file)
                    logger.info(f"✓ Model found: {model_file}")
                else:
                    models['missing_models'].append(model_file)
            
            # List all .pt files / 列出所有.pt文件
            all_models = list(models_dir.glob('*.pt'))
            if all_models:
                logger.info(f"Total model files found: {len(all_models)}")
            else:
                models['recommendations'].append("No trained models found. Run training script first.")
                logger.warning("⚠ No .pt model files found in models directory")
        else:
            models['models_directory_exists'] = False
            models['recommendations'].append("Models directory does not exist. Create it and add trained models.")
            logger.error("✗ Models directory not found")
        
        self.results['models'] = models
        return models
    
    def check_data(self) -> Dict[str, Any]:
        """Check data directory and samples."""
        logger.info("Checking data availability...")
        
        data = {
            'data_directory_exists': False,
            'sample_images_available': False,
            'datasets_available': [],
            'recommendations': []
        }
        
        data_dir = Path('data')
        if data_dir.exists():
            data['data_directory_exists'] = True
            
            # Check for sample images / 检查样本图像
            samples_dir = data_dir / 'samples'
            if samples_dir.exists() and list(samples_dir.glob('*.jpg')):
                data['sample_images_available'] = True
                logger.info("✓ Sample images found")
            else:
                data['recommendations'].append("Create sample images for testing")
                logger.warning("⚠ No sample images found")
            
            # Check for datasets / 检查数据集
            for subdir in data_dir.iterdir():
                if subdir.is_dir() and subdir.name != 'samples':
                    data['datasets_available'].append(subdir.name)
            
            if data['datasets_available']:
                logger.info(f"✓ Datasets found: {', '.join(data['datasets_available'])}")
            else:
                data['recommendations'].append("Generate or download datasets for training")
                logger.warning("⚠ No datasets found")
        else:
            data['data_directory_exists'] = False
            data['recommendations'].append("Create data directory structure")
            logger.error("✗ Data directory not found")
        
        self.results['data'] = data
        return data
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files."""
        logger.info("Checking configuration files...")
        
        config = {
            'config_directory_exists': False,
            'available_configs': [],
            'missing_configs': [],
            'recommendations': []
        }
        
        config_dir = Path('config')
        if config_dir.exists():
            config['config_directory_exists'] = True
            
            # Check for expected config files / 检查预期配置文件
            expected_configs = [
                'model_config.yaml',
                'camera_config.yaml'
            ]
            
            for config_file in expected_configs:
                config_path = config_dir / config_file
                if config_path.exists():
                    config['available_configs'].append(config_file)
                    logger.info(f"✓ Config found: {config_file}")
                else:
                    config['missing_configs'].append(config_file)
                    logger.warning(f"⚠ Config missing: {config_file}")
            
            if config['missing_configs']:
                config['recommendations'].append("Create missing configuration files")
        else:
            config['config_directory_exists'] = False
            config['recommendations'].append("Create config directory and configuration files")
            logger.error("✗ Config directory not found")
        
        self.results['configuration'] = config
        return config
    
    def run_performance_test(self) -> Dict[str, Any]:
        """Run basic performance test."""
        logger.info("Running performance test...")
        
        performance = {
            'test_completed': False,
            'import_time': 0,
            'model_load_time': 0,
            'inference_time': 0,
            'memory_usage_mb': 0,
            'recommendations': []
        }
        
        try:
            # Test imports / 测试导入
            start_time = time.time()
            from src.models.yolo_character_detector import YOLOCharacterDetector
            import_time = time.time() - start_time
            performance['import_time'] = import_time
            
            # Test model loading (if model available) / 测试模型加载（如果模型可用）
            models = self.results.get('models', {})
            if models.get('available_models'):
                try:
                    detector = YOLOCharacterDetector()
                    
                    start_time = time.time()
                    detector.load_model(pretrained=True)  # Use pretrained for testing
                    model_load_time = time.time() - start_time
                    performance['model_load_time'] = model_load_time
                    
                    # Test inference with dummy image / 使用虚拟图像测试推理
                    import numpy as np
                    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    
                    start_time = time.time()
                    detections = detector.predict(dummy_image)
                    inference_time = time.time() - start_time
                    performance['inference_time'] = inference_time
                    
                    performance['test_completed'] = True
                    
                except Exception as e:
                    performance['recommendations'].append(f"Performance test failed: {str(e)}")
                    logger.warning(f"Performance test failed: {e}")
            else:
                performance['recommendations'].append("No models available for performance testing")
            
            # Check memory usage / 检查内存使用
            memory_info = psutil.virtual_memory()
            performance['memory_usage_mb'] = round((memory_info.total - memory_info.available) / (1024**2), 1)
            
            # Performance recommendations / 性能建议
            if performance['inference_time'] > 0.1:  # > 100ms
                performance['recommendations'].append("Inference time is high. Consider model optimization.")
            
            if performance['memory_usage_mb'] > 3000:  # > 3GB
                performance['recommendations'].append("High memory usage detected. Monitor during operation.")
            
        except Exception as e:
            performance['recommendations'].append(f"Could not complete performance test: {str(e)}")
            logger.error(f"Performance test error: {e}")
        
        self.results['performance'] = performance
        return performance
    
    def determine_overall_status(self) -> str:
        """Determine overall deployment readiness status."""
        
        # Critical requirements / 关键要求
        critical_checks = [
            self.results['dependencies']['missing_required'] == [],
            self.results['hardware']['memory_sufficient'],
            self.results['hardware']['disk_space_sufficient'],
            self.results['hardware']['cpu_sufficient']
        ]
        
        # Important but not critical / 重要但非关键
        important_checks = [
            self.results['models']['models_directory_exists'],
            self.results['data']['data_directory_exists'],
            self.results['configuration']['config_directory_exists']
        ]
        
        # Determine status / 确定状态
        if all(critical_checks):
            if all(important_checks):
                status = "READY"
            else:
                status = "MOSTLY_READY"
        else:
            status = "NOT_READY"
        
        self.results['overall_status'] = status
        return status
    
    def run_complete_check(self) -> Dict[str, Any]:
        """Run complete deployment readiness check."""
        logger.info("Starting comprehensive deployment check...")
        
        # Run all checks / 运行所有检查
        self.check_system_info()
        self.check_dependencies()
        self.check_hardware_requirements()
        self.check_models()
        self.check_data()
        self.check_configuration()
        self.run_performance_test()
        
        # Determine overall status / 确定总体状态
        overall_status = self.determine_overall_status()
        
        # Add timestamp / 添加时间戳
        self.results['check_timestamp'] = time.time()
        
        return self.results
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate deployment readiness report."""
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("JETSON CHARACTER RECOGNITION - DEPLOYMENT CHECK")
        report_lines.append("="*60)
        report_lines.append(f"Overall Status: {self.results['overall_status']}")
        report_lines.append("")
        
        # System Information / 系统信息
        report_lines.append("SYSTEM INFORMATION:")
        system_info = self.results['system_info']
        report_lines.append(f"  Platform: {system_info['platform']}")
        report_lines.append(f"  Python: {system_info['python_version']}")
        report_lines.append(f"  Memory: {system_info['memory_available_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB")
        report_lines.append(f"  Jetson Device: {system_info['is_jetson']}")
        report_lines.append(f"  CUDA Available: {system_info.get('cuda_available', False)}")
        report_lines.append("")
        
        # Dependencies / 依赖项
        report_lines.append("DEPENDENCIES:")
        deps = self.results['dependencies']
        if deps['missing_required']:
            report_lines.append(f"  ✗ Missing required: {', '.join(deps['missing_required'])}")
        else:
            report_lines.append("  ✓ All required dependencies installed")
        
        if deps['missing_optional']:
            report_lines.append(f"  - Missing optional: {', '.join(deps['missing_optional'])}")
        report_lines.append("")
        
        # Hardware / 硬件
        report_lines.append("HARDWARE:")
        hardware = self.results['hardware']
        report_lines.append(f"  Memory: {'✓' if hardware['memory_sufficient'] else '✗'}")
        report_lines.append(f"  Disk Space: {'✓' if hardware['disk_space_sufficient'] else '✗'}")
        report_lines.append(f"  CPU: {'✓' if hardware['cpu_sufficient'] else '✗'}")
        report_lines.append(f"  GPU: {'✓' if hardware['gpu_available'] else '⚠'}")
        report_lines.append("")
        
        # Models / 模型
        report_lines.append("MODELS:")
        models = self.results['models']
        if models['available_models']:
            report_lines.append(f"  ✓ Available: {', '.join(models['available_models'])}")
        else:
            report_lines.append("  ✗ No models found")
        report_lines.append("")
        
        # Performance / 性能
        if self.results['performance']['test_completed']:
            perf = self.results['performance']
            report_lines.append("PERFORMANCE:")
            report_lines.append(f"  Model load time: {perf['model_load_time']:.3f}s")
            report_lines.append(f"  Inference time: {perf['inference_time']:.3f}s")
            report_lines.append(f"  Estimated FPS: {1/perf['inference_time']:.1f}" if perf['inference_time'] > 0 else "  Estimated FPS: N/A")
            report_lines.append("")
        
        # Recommendations / 建议
        all_recommendations = []
        for section in ['hardware', 'models', 'data', 'configuration', 'performance']:
            if section in self.results and 'recommendations' in self.results[section]:
                all_recommendations.extend(self.results[section]['recommendations'])
        
        if all_recommendations:
            report_lines.append("RECOMMENDATIONS:")
            for rec in all_recommendations:
                report_lines.append(f"  • {rec}")
            report_lines.append("")
        
        report_lines.append("="*60)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified / 如果指定则保存到文件
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_file}")
        
        return report_text


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check deployment readiness")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument("--report", help="Output file for human-readable report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Run deployment check / 运行部署检查
    checker = DeploymentChecker()
    results = checker.run_complete_check()
    
    # Generate and display report / 生成并显示报告
    report = checker.generate_report(args.report)
    print(report)
    
    # Save detailed results / 保存详细结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")
    
    # Exit with appropriate code / 使用适当的代码退出
    if results['overall_status'] == 'READY':
        print("\n🎉 System is ready for deployment!")
        sys.exit(0)
    elif results['overall_status'] == 'MOSTLY_READY':
        print("\n⚠️  System is mostly ready. Check recommendations above.")
        sys.exit(0)
    else:
        print("\n❌ System is not ready for deployment. Please address the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
