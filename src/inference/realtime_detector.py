"""
Real-time character detection pipeline.
"""

import cv2
import numpy as np
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..models.yolo_character_detector import YOLOCharacterDetector
from ..utils.logger import get_logger
from ..utils.performance import PerformanceMonitor, FPSCounter, PerformanceMetrics
from ..utils.config_loader import config_loader
from .camera_handler import CameraHandler

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Container for detection results."""
    frame: np.ndarray
    detections: List[Dict[str, Any]]
    timestamp: float
    processing_time: float
    fps: float


class RealtimeCharacterDetector:
    """Real-time character detection pipeline."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 camera_config: Optional[Dict[str, Any]] = None,
                 model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model
            camera_config: Camera configuration
            model_config: Model configuration
        """
        # Load configurations
        if camera_config is None:
            camera_config = config_loader.get_camera_config()
        if model_config is None:
            model_config = config_loader.get_model_config()
        
        self.camera_config = camera_config
        self.model_config = model_config
        
        # Initialize components
        self.camera = CameraHandler(camera_config)
        self.detector = YOLOCharacterDetector(model_config)
        self.performance_monitor = PerformanceMonitor()
        self.fps_counter = FPSCounter()
        
        # Load model
        if model_path:
            self.detector.load_model(model_path)
        else:
            logger.warning("No model path provided. Model must be loaded before detection.")
        
        # Detection settings
        self.confidence_threshold = model_config['model']['confidence_threshold']
        self.nms_threshold = model_config['model']['nms_threshold']
        
        # Display settings
        self.display_config = camera_config.get('display', {})
        self.show_preview = self.display_config.get('show_preview', True)
        self.show_fps = self.display_config.get('show_fps', True)
        self.show_detections = self.display_config.get('show_detections', True)
        
        # Recording settings
        self.recording_config = camera_config.get('recording', {})
        self.video_writer = None
        
        # State
        self.is_running = False
        self.detection_callback = None
        
        logger.info("Real-time character detector initialized")
    
    def set_detection_callback(self, callback: Callable[[DetectionResult], None]):
        """
        Set callback function for detection results.
        
        Args:
            callback: Function to call with detection results
        """
        self.detection_callback = callback
    
    def start_detection(self) -> bool:
        """
        Start real-time detection.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            logger.warning("Detection already running")
            return True
        
        # Start camera
        if not self.camera.start_capture():
            logger.error("Failed to start camera")
            return False
        
        # Initialize video recording if enabled
        if self.recording_config.get('enabled', False):
            self._initialize_recording()
        
        self.is_running = True
        
        # Start detection loop in separate thread
        detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        detection_thread.start()
        
        logger.info("Real-time detection started")
        return True
    
    def stop_detection(self):
        """Stop real-time detection."""
        self.is_running = False
        
        # Stop camera
        self.camera.stop_capture()
        
        # Stop recording
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Close display windows
        cv2.destroyAllWindows()
        
        logger.info("Real-time detection stopped")
    
    def _detection_loop(self):
        """Main detection loop."""
        logger.info("Starting detection loop")
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera.get_frame(timeout=1.0)
                if frame is None:
                    continue
                
                # Record timing
                start_time = time.time()
                
                # Preprocess frame
                self.performance_monitor.start_timer('preprocessing')
                processed_frame = self.camera.preprocess_frame(frame.copy())
                preprocessing_time = self.performance_monitor.stop_timer('preprocessing')
                
                # Run detection
                self.performance_monitor.start_timer('inference')
                detections = self.detector.predict(
                    processed_frame,
                    confidence=self.confidence_threshold,
                    nms_threshold=self.nms_threshold
                )
                inference_time = self.performance_monitor.stop_timer('inference')
                
                # Post-process results
                self.performance_monitor.start_timer('postprocessing')
                result_frame = self._draw_detections(frame, detections)
                postprocessing_time = self.performance_monitor.stop_timer('postprocessing')
                
                # Calculate timing metrics
                total_time = time.time() - start_time
                fps = self.fps_counter.update()
                
                # Record performance metrics
                metrics = PerformanceMetrics(
                    inference_time=inference_time,
                    preprocessing_time=preprocessing_time,
                    postprocessing_time=postprocessing_time,
                    total_time=total_time,
                    fps=fps
                )
                self.performance_monitor.record_metrics(metrics)
                
                # Create detection result
                detection_result = DetectionResult(
                    frame=result_frame,
                    detections=detections,
                    timestamp=time.time(),
                    processing_time=total_time,
                    fps=fps
                )
                
                # Call detection callback if set
                if self.detection_callback:
                    try:
                        self.detection_callback(detection_result)
                    except Exception as e:
                        logger.error(f"Error in detection callback: {e}")
                
                # Display frame
                if self.show_preview:
                    self._display_frame(result_frame, fps, detections)
                
                # Record frame
                if self.video_writer:
                    self.video_writer.write(result_frame)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(0.1)
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        if not self.show_detections:
            return frame
        
        result_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(result_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         (0, 255, 0), -1)
            
            # Label text
            cv2.putText(result_frame, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_frame
    
    def _display_frame(self, frame: np.ndarray, fps: float, detections: List[Dict[str, Any]]):
        """
        Display frame with overlays.
        
        Args:
            frame: Frame to display
            fps: Current FPS
            detections: Detection results
        """
        display_frame = frame.copy()
        
        # Add FPS overlay
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection count
        if self.show_detections:
            count_text = f"Detections: {len(detections)}"
            cv2.putText(display_frame, count_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize for display if needed
        preview_width = self.display_config.get('preview_width', 640)
        preview_height = self.display_config.get('preview_height', 480)
        
        if display_frame.shape[1] != preview_width or display_frame.shape[0] != preview_height:
            display_frame = cv2.resize(display_frame, (preview_width, preview_height))
        
        # Show frame
        cv2.imshow('Character Detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop_detection()
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f"detection_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Saved frame: {filename}")
    
    def _initialize_recording(self):
        """Initialize video recording."""
        try:
            output_dir = self.recording_config.get('output_dir', 'recordings')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"detection_recording_{timestamp}.{self.recording_config.get('format', 'mp4')}"
            output_path = Path(output_dir) / filename
            
            # Video codec
            codec = self.recording_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Video properties
            fps = self.camera.fps
            width = self.camera.frame_width
            height = self.camera.frame_height
            
            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
            
            logger.info(f"Recording initialized: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize recording: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_monitor.get_performance_summary()
    
    def detect_single_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect characters in a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Detection result
        """
        start_time = time.time()
        
        # Preprocess
        processed_frame = self.camera.preprocess_frame(frame.copy())
        
        # Detect
        detections = self.detector.predict(
            processed_frame,
            confidence=self.confidence_threshold,
            nms_threshold=self.nms_threshold
        )
        
        # Draw results
        result_frame = self._draw_detections(frame, detections)
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            frame=result_frame,
            detections=detections,
            timestamp=time.time(),
            processing_time=processing_time,
            fps=1.0 / processing_time if processing_time > 0 else 0
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.start_detection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_detection()
