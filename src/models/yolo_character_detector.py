"""
YOLOv8-based character detection and recognition model.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import yaml

from ..utils.logger import get_logger
from ..utils.config_loader import config_loader

logger = get_logger(__name__)


class YOLOCharacterDetector:
    """YOLOv8-based character detector optimized for Jetson Nano."""
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        Initialize the character detector.
        
        Args:
            model_config: Model configuration dictionary
        """
        if model_config is None:
            model_config = config_loader.get_model_config()
        
        self.config = model_config
        self.model = None
        self.device = self._get_device()
        
        # Character classes (0-9, A-Z)
        self.digit_classes = [str(i) for i in range(10)]
        self.letter_classes = [chr(ord('A') + i) for i in range(26)]
        self.all_classes = self.digit_classes + self.letter_classes
        self.class_names = {i: cls for i, cls in enumerate(self.all_classes)}
        
        logger.info(f"Initialized YOLOCharacterDetector with {len(self.all_classes)} classes")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_model(self, model_path: Optional[str] = None, pretrained: bool = True) -> None:
        """
        Load or create the YOLO model.
        
        Args:
            model_path: Path to trained model weights
            pretrained: Whether to use pretrained weights
        """
        try:
            if model_path and Path(model_path).exists():
                logger.info(f"Loading model from {model_path}")
                self.model = YOLO(model_path)
            else:
                # Create new model
                model_name = self.config['model']['name']
                logger.info(f"Creating new {model_name} model")
                
                if pretrained:
                    self.model = YOLO(f"{model_name}.pt")
                else:
                    self.model = YOLO(f"{model_name}.yaml")
            
            # Move to device
            if self.device == "cuda":
                self.model.to(self.device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(self, dataset_path: str, epochs: int = None, **kwargs) -> Dict[str, Any]:
        """
        Train the character detection model.
        
        Args:
            dataset_path: Path to YOLO format dataset
            epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        if self.model is None:
            self.load_model(pretrained=True)
        
        # Get training parameters
        train_config = self.config.get('training', {})
        epochs = epochs or train_config.get('epochs', 100)
        batch_size = kwargs.get('batch_size', train_config.get('batch_size', 16))
        learning_rate = kwargs.get('lr', train_config.get('learning_rate', 0.001))
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        try:
            # Train the model
            results = self.model.train(
                data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                device=self.device,
                project="runs/train",
                name="character_detection",
                exist_ok=True,
                **kwargs
            )
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, image: np.ndarray, 
                confidence: float = None,
                nms_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Predict characters in an image.
        
        Args:
            image: Input image as numpy array
            confidence: Confidence threshold
            nms_threshold: NMS threshold
            
        Returns:
            List of detected characters with bounding boxes and classes
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not provided
        confidence = confidence or self.config['model']['confidence_threshold']
        nms_threshold = nms_threshold or self.config['model']['nms_threshold']
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=confidence,
                iou=nms_threshold,
                device=self.device,
                verbose=False
            )
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if cls_id < len(self.all_classes):
                            detection = {
                                'bbox': box.tolist(),  # [x1, y1, x2, y2]
                                'confidence': float(conf),
                                'class_id': int(cls_id),
                                'class_name': self.all_classes[cls_id],
                                'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                                'width': box[2] - box[0],
                                'height': box[3] - box[1]
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
    
    def predict_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict[str, Any]]]:
        """
        Predict characters in a batch of images.
        
        Args:
            images: List of input images
            **kwargs: Prediction parameters
            
        Returns:
            List of detection results for each image
        """
        results = []
        for image in images:
            detections = self.predict(image, **kwargs)
            results.append(detections)
        return results
    
    def export_model(self, output_path: str, format: str = "onnx") -> str:
        """
        Export model to different formats for deployment.
        
        Args:
            output_path: Output file path
            format: Export format (onnx, tensorrt, etc.)
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Exporting model to {format} format")
        
        try:
            exported_path = self.model.export(
                format=format,
                device=self.device,
                half=self.config.get('jetson', {}).get('fp16', True)
            )
            
            logger.info(f"Model exported to: {exported_path}")
            return exported_path
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise
    
    def optimize_for_jetson(self, model_path: str) -> str:
        """
        Optimize model for Jetson Nano deployment.
        
        Args:
            model_path: Path to the model to optimize
            
        Returns:
            Path to optimized model
        """
        logger.info("Optimizing model for Jetson Nano")
        
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model(model_path)
            
            # Export to TensorRT format for Jetson optimization
            jetson_config = self.config.get('jetson', {})
            
            if jetson_config.get('use_tensorrt', True):
                optimized_path = self.export_model(
                    model_path.replace('.pt', '_tensorrt.engine'),
                    format='engine'
                )
            else:
                # Export to ONNX with optimization
                optimized_path = self.export_model(
                    model_path.replace('.pt', '_optimized.onnx'),
                    format='onnx'
                )
            
            logger.info(f"Model optimized for Jetson: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            logger.error(f"Jetson optimization failed: {e}")
            raise
    
    def validate(self, dataset_path: str) -> Dict[str, float]:
        """
        Validate the model on a dataset.
        
        Args:
            dataset_path: Path to validation dataset
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Running model validation")
        
        try:
            results = self.model.val(
                data=dataset_path,
                device=self.device,
                project="runs/val",
                name="character_detection",
                exist_ok=True
            )
            
            # Extract key metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
            }
            
            logger.info(f"Validation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        try:
            info = {
                "model_type": "YOLOv8",
                "num_classes": len(self.all_classes),
                "classes": self.all_classes,
                "device": self.device,
                "input_size": self.config['model']['input_size'],
                "confidence_threshold": self.config['model']['confidence_threshold'],
                "nms_threshold": self.config['model']['nms_threshold']
            }
            
            # Add model-specific info if available
            if hasattr(self.model, 'model'):
                info["parameters"] = sum(p.numel() for p in self.model.model.parameters())
                info["trainable_parameters"] = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def save_model(self, output_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            torch.save(self.model.model.state_dict(), output_path)
            logger.info(f"Model saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
