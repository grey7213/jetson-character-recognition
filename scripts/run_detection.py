#!/usr/bin/env python3
"""
Run real-time character detection.
"""

import argparse
import sys
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.realtime_detector import RealtimeCharacterDetector
from src.utils.logger import setup_logger
from src.utils.jetson_utils import jetson_monitor, jetson_optimizer

logger = setup_logger("run_detection", level="INFO")


class DetectionRunner:
    """Run real-time character detection."""
    
    def __init__(self, model_path: str, optimize_jetson: bool = True):
        """
        Initialize detection runner.
        
        Args:
            model_path: Path to trained model
            optimize_jetson: Whether to apply Jetson optimizations
        """
        self.model_path = model_path
        self.detector = None
        self.running = False
        
        # Apply Jetson optimizations if requested
        if optimize_jetson and jetson_monitor.is_jetson:
            logger.info("Applying Jetson optimizations")
            jetson_optimizer.setup_environment()
            jetson_optimizer.optimize_opencv()
            jetson_monitor.optimize_performance()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def detection_callback(self, result):
        """
        Callback for detection results.
        
        Args:
            result: Detection result
        """
        # Print detection summary
        if result.detections:
            characters = [d['class_name'] for d in result.detections]
            confidences = [d['confidence'] for d in result.detections]
            
            logger.info(f"Detected: {characters} (conf: {[f'{c:.2f}' for c in confidences]})")
    
    def start(self):
        """Start real-time detection."""
        try:
            logger.info("Initializing real-time character detector")
            
            # Initialize detector
            self.detector = RealtimeCharacterDetector(model_path=self.model_path)
            self.detector.set_detection_callback(self.detection_callback)
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Start detection
            logger.info("Starting real-time detection")
            logger.info("Press 'q' in the preview window or Ctrl+C to stop")
            
            if not self.detector.start_detection():
                logger.error("Failed to start detection")
                return False
            
            self.running = True
            
            # Keep running until stopped
            try:
                while self.running:
                    import time
                    time.sleep(1)
                    
                    # Print performance summary periodically
                    if hasattr(self, '_last_summary_time'):
                        if time.time() - self._last_summary_time > 30:  # Every 30 seconds
                            self._print_performance_summary()
                            self._last_summary_time = time.time()
                    else:
                        self._last_summary_time = time.time()
            
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            
            return True
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return False
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop detection."""
        self.running = False
        
        if self.detector:
            logger.info("Stopping detection")
            self.detector.stop_detection()
            
            # Print final performance summary
            self._print_performance_summary()
    
    def _print_performance_summary(self):
        """Print performance summary."""
        if not self.detector:
            return
        
        try:
            summary = self.detector.get_performance_summary()
            
            if 'recent_performance' in summary:
                recent = summary['recent_performance']
                logger.info(f"Performance - FPS: {recent.get('fps', 0):.1f}, "
                           f"Inference: {recent.get('inference_time', 0)*1000:.1f}ms")
        
        except Exception as e:
            logger.warning(f"Failed to get performance summary: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Real-time Character Detection")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("--no-jetson-optimization", action="store_true",
                       help="Disable Jetson-specific optimizations")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Print system info
    system_info = jetson_monitor.get_system_info()
    logger.info(f"System info: {system_info}")
    
    # Run detection
    runner = DetectionRunner(
        str(model_path), 
        optimize_jetson=not args.no_jetson_optimization
    )
    
    success = runner.start()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
