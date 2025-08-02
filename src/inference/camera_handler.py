"""
Camera handling for real-time character detection.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Dict, Any, Callable
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class CameraHandler:
    """Handle camera input for real-time processing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize camera handler.
        
        Args:
            config: Camera configuration
        """
        if config is None:
            config = config_loader.get_camera_config()
        
        self.config = config
        self.camera_config = config.get('camera', {})
        self.capture_config = config.get('capture', {})
        self.preprocessing_config = config.get('preprocessing', {})
        
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=self.config.get('performance', {}).get('queue_size', 2))
        self.capture_thread = None
        
        # Camera properties
        self.camera_type = self.camera_config.get('type', 'usb')
        self.device_id = self.camera_config.get('device_id', 0)
        
        # Frame properties
        self.frame_width = self.capture_config.get('width', 1280)
        self.frame_height = self.capture_config.get('height', 720)
        self.fps = self.capture_config.get('fps', 30)
        
        logger.info(f"Initialized camera handler: {self.camera_type}")
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.camera_type == 'usb':
                self.cap = cv2.VideoCapture(self.device_id)
            elif self.camera_type == 'csi':
                # CSI camera pipeline for Jetson
                csi_config = self.camera_config.get('csi', {})
                sensor_id = csi_config.get('sensor_id', 0)
                sensor_mode = csi_config.get('sensor_mode', 0)
                flip_method = csi_config.get('flip_method', 0)
                
                gstreamer_pipeline = (
                    f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
                    f"video/x-raw(memory:NVMM), width={self.frame_width}, height={self.frame_height}, "
                    f"format=NV12, framerate={self.fps}/1 ! "
                    f"nvvidconv flip-method={flip_method} ! "
                    f"video/x-raw, width={self.frame_width}, height={self.frame_height}, format=BGRx ! "
                    f"videoconvert ! "
                    f"video/x-raw, format=BGR ! appsink"
                )
                
                self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
                
            elif self.camera_type == 'ip':
                # IP camera
                ip_config = self.camera_config.get('ip', {})
                url = ip_config.get('url', '')
                self.cap = cv2.VideoCapture(url)
            
            else:
                raise ValueError(f"Unsupported camera type: {self.camera_type}")
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Reduce buffer size for lower latency
            buffer_size = self.capture_config.get('buffer_size', 1)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def start_capture(self) -> bool:
        """
        Start camera capture in a separate thread.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            logger.warning("Camera capture already running")
            return True
        
        if self.cap is None:
            if not self.initialize_camera():
                return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Camera capture started")
        return True
    
    def stop_capture(self):
        """Stop camera capture."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Camera capture stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        timeout = self.config.get('performance', {}).get('timeout', 5.0)
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the latest frame from camera.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame as numpy array or None if timeout
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Resize frame
        target_width = self.preprocessing_config.get('resize_width', 640)
        target_height = self.preprocessing_config.get('resize_height', 640)
        
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))
        
        # Normalize if requested
        if self.preprocessing_config.get('normalize', False):
            frame = frame.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array(self.preprocessing_config.get('mean', [0.485, 0.456, 0.406]))
            std = np.array(self.preprocessing_config.get('std', [0.229, 0.224, 0.225]))
            
            frame = (frame - mean) / std
        
        return frame
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        if self.cap is None:
            return {'status': 'Camera not initialized'}
        
        try:
            info = {
                'camera_type': self.camera_type,
                'device_id': self.device_id,
                'is_running': self.is_running,
                'frame_width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'frame_height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'backend': self.cap.getBackendName(),
                'queue_size': self.frame_queue.qsize()
            }
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_camera(self, duration: float = 5.0) -> Dict[str, Any]:
        """
        Test camera functionality.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Test results
        """
        logger.info(f"Testing camera for {duration} seconds...")
        
        if not self.initialize_camera():
            return {'status': 'failed', 'error': 'Failed to initialize camera'}
        
        start_time = time.time()
        frame_count = 0
        successful_frames = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = self.cap.read()
                frame_count += 1
                
                if ret and frame is not None:
                    successful_frames += 1
                
                time.sleep(0.033)  # ~30 FPS
            
            success_rate = (successful_frames / frame_count) * 100 if frame_count > 0 else 0
            actual_fps = frame_count / duration
            
            results = {
                'status': 'success',
                'duration': duration,
                'total_frames': frame_count,
                'successful_frames': successful_frames,
                'success_rate': success_rate,
                'actual_fps': actual_fps,
                'camera_info': self.get_camera_info()
            }
            
            logger.info(f"Camera test completed: {success_rate:.1f}% success rate, {actual_fps:.1f} FPS")
            return results
            
        except Exception as e:
            logger.error(f"Camera test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
        
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def save_test_frames(self, output_dir: str, num_frames: int = 10) -> List[str]:
        """
        Save test frames from camera.
        
        Args:
            output_dir: Output directory
            num_frames: Number of frames to save
            
        Returns:
            List of saved frame paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.initialize_camera():
            logger.error("Failed to initialize camera for test frames")
            return []
        
        saved_frames = []
        
        try:
            for i in range(num_frames):
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    frame_path = output_path / f"test_frame_{i:03d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    saved_frames.append(str(frame_path))
                    logger.info(f"Saved test frame: {frame_path}")
                
                time.sleep(0.5)  # Wait between frames
            
            return saved_frames
            
        except Exception as e:
            logger.error(f"Failed to save test frames: {e}")
            return saved_frames
        
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()
