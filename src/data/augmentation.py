"""
Data augmentation for character recognition training.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Tuple, List
import random

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CharacterAugmentation:
    """Data augmentation pipeline for character recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if self.enabled:
            self.transform = self._create_augmentation_pipeline()
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create the augmentation pipeline."""
        transforms = []
        
        # Geometric transformations
        if self.config.get('rotation', 0) > 0:
            transforms.append(
                A.Rotate(
                    limit=self.config['rotation'],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                )
            )
        
        # Scale transformations
        if 'scale' in self.config:
            scale_range = self.config['scale']
            transforms.append(
                A.RandomScale(
                    scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0),
                    p=0.5
                )
            )
        
        # Perspective transformation
        transforms.append(
            A.Perspective(
                scale=(0.02, 0.05),
                p=0.3
            )
        )
        
        # Elastic transformation for slight deformation
        transforms.append(
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.2
            )
        )
        
        # Color/brightness transformations
        color_transforms = []
        
        if self.config.get('brightness', 0) > 0:
            color_transforms.append(
                A.RandomBrightness(
                    limit=self.config['brightness'],
                    p=0.5
                )
            )
        
        if self.config.get('contrast', 0) > 0:
            color_transforms.append(
                A.RandomContrast(
                    limit=self.config['contrast'],
                    p=0.5
                )
            )
        
        if color_transforms:
            transforms.append(A.OneOf(color_transforms, p=0.7))
        
        # Noise and blur
        noise_transforms = [
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3)
        ]
        transforms.append(A.OneOf(noise_transforms, p=0.4))
        
        blur_transforms = [
            A.Blur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3)
        ]
        transforms.append(A.OneOf(blur_transforms, p=0.3))
        
        # Cutout for robustness
        transforms.append(
            A.CoarseDropout(
                max_holes=3,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=4,
                min_width=4,
                fill_value=0,
                p=0.2
            )
        )
        
        # Final normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def __call__(self, image: np.ndarray, bboxes: List = None, 
                 class_labels: List = None) -> Dict[str, Any]:
        """
        Apply augmentation to image and annotations.
        
        Args:
            image: Input image
            bboxes: Bounding boxes (optional)
            class_labels: Class labels (optional)
            
        Returns:
            Augmented image and annotations
        """
        if not self.enabled:
            return {
                'image': self.transform(image=image)['image'],
                'bboxes': bboxes,
                'class_labels': class_labels
            }
        
        # Prepare albumentations format
        if bboxes is not None and class_labels is not None:
            # Convert to albumentations format if needed
            transform_with_bbox = A.Compose(
                self.transform.transforms[:-2],  # Exclude normalization and tensor conversion
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels']
                )
            )
            
            result = transform_with_bbox(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Apply final normalization and tensor conversion
            final_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            final_result = final_transform(image=result['image'])
            
            return {
                'image': final_result['image'],
                'bboxes': result['bboxes'],
                'class_labels': result['class_labels']
            }
        else:
            result = self.transform(image=image)
            return {
                'image': result['image'],
                'bboxes': bboxes,
                'class_labels': class_labels
            }
    
    def create_validation_transform(self) -> A.Compose:
        """Create validation transform (no augmentation)."""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class CharacterMixUp:
    """MixUp augmentation for character recognition."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying MixUp
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_images: np.ndarray, batch_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply MixUp to a batch of images and labels.
        
        Args:
            batch_images: Batch of images
            batch_labels: Batch of labels
            
        Returns:
            Mixed images and labels
        """
        if random.random() > self.prob:
            return batch_images, batch_labels
        
        batch_size = batch_images.shape[0]
        
        # Generate mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Generate random permutation
        index = np.random.permutation(batch_size)
        
        # Mix images
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        
        # Mix labels (for classification)
        if len(batch_labels.shape) == 1:
            # One-hot encode labels first
            num_classes = len(np.unique(batch_labels))
            y_a = np.eye(num_classes)[batch_labels]
            y_b = np.eye(num_classes)[batch_labels[index]]
            mixed_labels = lam * y_a + (1 - lam) * y_b
        else:
            mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index]
        
        return mixed_images, mixed_labels


class CharacterCutMix:
    """CutMix augmentation for character recognition."""
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize CutMix augmentation.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_images: np.ndarray, batch_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix to a batch of images and labels.
        
        Args:
            batch_images: Batch of images
            batch_labels: Batch of labels
            
        Returns:
            CutMix images and labels
        """
        if random.random() > self.prob:
            return batch_images, batch_labels
        
        batch_size = batch_images.shape[0]
        
        # Generate mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Generate random permutation
        index = np.random.permutation(batch_size)
        
        # Generate bounding box
        W, H = batch_images.shape[2], batch_images.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = batch_images.copy()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = batch_images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Mix labels
        if len(batch_labels.shape) == 1:
            # One-hot encode labels first
            num_classes = len(np.unique(batch_labels))
            y_a = np.eye(num_classes)[batch_labels]
            y_b = np.eye(num_classes)[batch_labels[index]]
            mixed_labels = lam * y_a + (1 - lam) * y_b
        else:
            mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index]
        
        return mixed_images, mixed_labels
