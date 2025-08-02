"""
Jetson Character Recognition System

A computer vision system for real-time character detection and recognition
optimized for NVIDIA Jetson Nano hardware.
"""

__version__ = "1.0.0"
__author__ = "Jetson Character Recognition Team"
__email__ = "support@example.com"

# Import main classes for easy access
from .models.yolo_character_detector import YOLOCharacterDetector
from .inference.realtime_detector import RealtimeCharacterDetector
from .data.dataset_manager import DatasetManager

__all__ = [
    "YOLOCharacterDetector",
    "RealtimeCharacterDetector", 
    "DatasetManager",
]
