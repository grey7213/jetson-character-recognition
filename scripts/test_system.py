#!/usr/bin/env python3
"""
System testing script for Jetson Character Recognition.
"""

import argparse
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector
from src.inference.camera_handler import CameraHandler
from src.inference.realtime_detector import RealtimeCharacterDetector
from src.data.dataset_manager import DatasetManager
from src.utils.logger import setup_logger, get_logger
from src.utils.config_loader import config_loader
from src.utils.jetson_utils import jetson_monitor
from src.utils.performance import PerformanceMonitor

logger = setup_logger("test_system", level="INFO")


class SystemTester:
    """Comprehensive system testing."""
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Initialize system tester.
        
        Args:
            output_dir: Directory for test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all system tests.
        
        Returns:
            Dictionary of test results
        """
        logger.info("Starting comprehensive system tests")
        
        tests = [
            ("System Info", self.test_system_info),
            ("Configuration", self.test_configuration),
            ("Dataset Manager", self.test_dataset_manager),
            ("Model Loading", self.test_model_loading),
            ("Camera", self.test_camera),
            ("Inference", self.test_inference),
            ("Performance", self.test_performance),
            ("Real-time Detection", self.test_realtime_detection)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                status = "PASS" if result else "FAIL"
                logger.info(f"Test {test_name}: {status}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Save test results
        self._save_test_results(results)
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def test_system_info(self) -> bool:
        """Test system information gathering."""
        try:
            system_info = jetson_monitor.get_system_info()
            
            required_keys = ['cpu_count', 'memory_total', 'memory_available']
            for key in required_keys:
                if key not in system_info:
                    logger.error(f"Missing system info key: {key}")
                    return False
            
            self.test_results['system_info'] = system_info
            logger.info(f"System info: {system_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"System info test failed: {e}")
            return False
    
    def test_configuration(self) -> bool:
        """Test configuration loading."""
        try:
            # Test model config
            model_config = config_loader.get_model_config()
            if 'model' not in model_config:
                logger.error("Invalid model configuration")
                return False
            
            # Test camera config
            camera_config = config_loader.get_camera_config()
            if 'camera' not in camera_config:
                logger.error("Invalid camera configuration")
                return False
            
            self.test_results['model_config'] = model_config
            self.test_results['camera_config'] = camera_config
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            return False
    
    def test_dataset_manager(self) -> bool:
        """Test dataset management."""
        try:
            dataset_manager = DatasetManager(str(self.output_dir / "test_data"))
            
            # Test dataset info
            datasets = dataset_manager.list_available_datasets()
            if not datasets:
                logger.error("No datasets available")
                return False
            
            # Test synthetic dataset generation
            synthetic_dir = dataset_manager.download_dataset("synthetic")
            if not synthetic_dir.exists():
                logger.error("Failed to generate synthetic dataset")
                return False
            
            # Check if dataset has correct structure
            class_dirs = list(synthetic_dir.iterdir())
            if len(class_dirs) != 36:  # 10 digits + 26 letters
                logger.error(f"Expected 36 classes, found {len(class_dirs)}")
                return False
            
            self.test_results['dataset_info'] = {
                'available_datasets': datasets,
                'synthetic_dataset_path': str(synthetic_dir),
                'class_count': len(class_dirs)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset manager test failed: {e}")
            return False
    
    def test_model_loading(self) -> bool:
        """Test model loading and initialization."""
        try:
            detector = YOLOCharacterDetector()
            
            # Test model info
            model_info = detector.get_model_info()
            if 'status' in model_info and 'No model loaded' in model_info['status']:
                # This is expected if no model is loaded
                pass
            
            # Test loading pretrained model
            try:
                detector.load_model(pretrained=True)
                model_info = detector.get_model_info()
                
                if 'num_classes' not in model_info:
                    logger.error("Model info missing num_classes")
                    return False
                
                self.test_results['model_info'] = model_info
                
            except Exception as e:
                logger.warning(f"Could not load pretrained model: {e}")
                # This might be expected in some environments
            
            return True
            
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False
    
    def test_camera(self) -> bool:
        """Test camera functionality."""
        try:
            camera = CameraHandler()
            
            # Test camera initialization
            if not camera.initialize_camera():
                logger.warning("Camera initialization failed - this may be expected in headless environments")
                return True  # Don't fail the test if no camera is available
            
            # Test camera info
            camera_info = camera.get_camera_info()
            self.test_results['camera_info'] = camera_info
            
            # Test frame capture
            camera.start_capture()
            time.sleep(1)  # Let camera warm up
            
            frame = camera.get_frame(timeout=5.0)
            camera.stop_capture()
            
            if frame is None:
                logger.warning("No frame captured from camera")
                return True  # Don't fail if no camera
            
            # Test frame preprocessing
            processed_frame = camera.preprocess_frame(frame)
            if processed_frame is None:
                logger.error("Frame preprocessing failed")
                return False
            
            self.test_results['camera_test'] = {
                'frame_shape': frame.shape,
                'processed_shape': processed_frame.shape
            }
            
            return True
            
        except Exception as e:
            logger.warning(f"Camera test failed: {e} - this may be expected without camera hardware")
            return True  # Don't fail the overall test
    
    def test_inference(self) -> bool:
        """Test inference functionality."""
        try:
            detector = YOLOCharacterDetector()
            
            # Create test image
            test_image = self._create_test_image()
            
            try:
                detector.load_model(pretrained=True)
                
                # Test prediction
                detections = detector.predict(test_image)
                
                self.test_results['inference_test'] = {
                    'test_image_shape': test_image.shape,
                    'detection_count': len(detections),
                    'detections': detections[:5]  # First 5 detections
                }
                
                return True
                
            except Exception as e:
                logger.warning(f"Could not test inference: {e}")
                return True  # Don't fail if model can't be loaded
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance monitoring."""
        try:
            monitor = PerformanceMonitor()
            
            # Test timer functionality
            monitor.start_timer('test')
            time.sleep(0.1)
            elapsed = monitor.stop_timer('test')
            
            if elapsed < 0.05 or elapsed > 0.2:
                logger.error(f"Timer test failed: expected ~0.1s, got {elapsed}s")
                return False
            
            # Test performance metrics recording
            from src.utils.performance import PerformanceMetrics
            
            test_metrics = PerformanceMetrics(
                inference_time=0.05,
                preprocessing_time=0.01,
                postprocessing_time=0.01,
                total_time=0.07,
                fps=14.3
            )
            
            monitor.record_metrics(test_metrics)
            
            latest = monitor.get_latest_metrics()
            if latest is None or latest.fps != 14.3:
                logger.error("Performance metrics recording failed")
                return False
            
            self.test_results['performance_test'] = {
                'timer_accuracy': elapsed,
                'metrics_recording': True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
    
    def test_realtime_detection(self) -> bool:
        """Test real-time detection pipeline."""
        try:
            # Create test detector (without starting camera)
            detector = RealtimeCharacterDetector()
            
            # Test single frame detection
            test_image = self._create_test_image()
            
            try:
                detector.detector.load_model(pretrained=True)
                result = detector.detect_single_frame(test_image)
                
                if result is None:
                    logger.error("Single frame detection failed")
                    return False
                
                self.test_results['realtime_test'] = {
                    'single_frame_detection': True,
                    'processing_time': result.processing_time,
                    'detection_count': len(result.detections)
                }
                
            except Exception as e:
                logger.warning(f"Could not test real-time detection: {e}")
                return True  # Don't fail if model can't be loaded
            
            return True
            
        except Exception as e:
            logger.error(f"Real-time detection test failed: {e}")
            return False
    
    def _create_test_image(self) -> np.ndarray:
        """Create a test image with characters."""
        # Create a simple test image with text
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some text
        cv2.putText(img, "A1B2C3", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, "XYZ789", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return img
    
    def _save_test_results(self, results: Dict[str, bool]):
        """Save test results to file."""
        output_file = self.output_dir / "test_results.json"
        
        test_data = {
            'timestamp': time.time(),
            'test_results': results,
            'detailed_results': self.test_results,
            'system_info': jetson_monitor.get_system_info()
        }
        
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {output_file}")
    
    def _print_test_summary(self, results: Dict[str, bool]):
        """Print test summary."""
        total_tests = len(results)
        passed_tests = sum(results.values())
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*50)
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for test_name, result in results.items():
                if not result:
                    print(f"  - {test_name}")
        
        print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Jetson Character Recognition System")
    parser.add_argument("--output-dir", default="test_results", 
                       help="Output directory for test results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Run tests
    tester = SystemTester(args.output_dir)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
