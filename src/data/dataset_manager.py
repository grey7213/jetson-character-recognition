"""
Dataset management for character recognition training.
"""

import os
import cv2
import numpy as np
import requests
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse
import shutil

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatasetManager:
    """Manages datasets for character recognition training."""
    
    # Popular character recognition datasets
    DATASETS = {
        "emnist": {
            "name": "EMNIST (Extended MNIST)",
            "url": "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
            "description": "Extended MNIST with letters and digits",
            "classes": 47,  # 10 digits + 26 letters + 11 additional
            "format": "numpy"
        },
        "chars74k": {
            "name": "Chars74K",
            "url": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz",
            "description": "Natural images of characters",
            "classes": 62,  # 10 digits + 26 lowercase + 26 uppercase
            "format": "images"
        },
        "synthetic": {
            "name": "Synthetic Character Dataset",
            "description": "Generated synthetic characters",
            "classes": 36,  # 10 digits + 26 letters
            "format": "generated"
        }
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Character classes (0-9, A-Z)
        self.digit_classes = [str(i) for i in range(10)]
        self.letter_classes = [chr(ord('A') + i) for i in range(26)]
        self.all_classes = self.digit_classes + self.letter_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.all_classes)}
        
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> Path:
        """
        Download a dataset if not already present.
        
        Args:
            dataset_name: Name of the dataset to download
            force_download: Force re-download even if exists
            
        Returns:
            Path to the downloaded dataset directory
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.DATASETS[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        
        if dataset_dir.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists at {dataset_dir}")
            return dataset_dir
        
        if dataset_name == "synthetic":
            return self._generate_synthetic_dataset()
        
        logger.info(f"Downloading {dataset_info['name']}...")
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download file
        url = dataset_info["url"]
        filename = Path(urlparse(url).path).name
        download_path = dataset_dir / filename
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {filename}")
            
            # Extract if it's an archive
            if filename.endswith(('.zip', '.tgz', '.tar.gz')):
                self._extract_archive(download_path, dataset_dir)
                
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise
        
        return dataset_dir
    
    def _extract_archive(self, archive_path: Path, extract_dir: Path):
        """Extract archive file."""
        logger.info(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in ['.tgz', '.gz']:
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        # Remove the archive file after extraction
        archive_path.unlink()
        logger.info("Extraction completed")
    
    def _generate_synthetic_dataset(self) -> Path:
        """Generate a synthetic character dataset."""
        logger.info("Generating synthetic character dataset...")
        
        dataset_dir = self.data_dir / "synthetic"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images for each character class
        for class_name in self.all_classes:
            class_dir = dataset_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Generate multiple variations of each character
            for i in range(100):  # 100 samples per class
                img = self._generate_character_image(class_name, variation=i)
                img_path = class_dir / f"{class_name}_{i:03d}.png"
                cv2.imwrite(str(img_path), img)
        
        logger.info(f"Generated synthetic dataset with {len(self.all_classes)} classes")
        return dataset_dir
    
    def _generate_character_image(self, character: str, variation: int = 0) -> np.ndarray:
        """
        Generate a synthetic character image with variations.
        
        Args:
            character: Character to generate
            variation: Variation number for different styles
            
        Returns:
            Generated image as numpy array
        """
        # Image size
        img_size = (64, 64)
        img = np.zeros(img_size, dtype=np.uint8)
        
        # Font variations
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX
        ]
        
        font = fonts[variation % len(fonts)]
        font_scale = 1.5 + (variation % 3) * 0.3  # Vary font size
        thickness = 2 + (variation % 2)  # Vary thickness
        
        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(
            character, font, font_scale, thickness
        )
        
        # Center the text
        x = (img_size[1] - text_width) // 2
        y = (img_size[0] + text_height) // 2
        
        # Add some random offset for variation
        x += (variation % 7) - 3
        y += (variation % 7) - 3
        
        # Draw the character
        cv2.putText(img, character, (x, y), font, font_scale, 255, thickness)
        
        # Add some noise and transformations for robustness
        if variation % 4 == 0:
            # Add Gaussian noise
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        if variation % 5 == 0:
            # Add slight rotation
            angle = (variation % 21) - 10  # -10 to +10 degrees
            center = (img_size[1] // 2, img_size[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, img_size)
        
        return img
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset."""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self.DATASETS[dataset_name].copy()
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.DATASETS.keys())
    
    def prepare_yolo_dataset(self, dataset_dir: Path, output_dir: Path):
        """
        Convert dataset to YOLO format for training.
        
        Args:
            dataset_dir: Source dataset directory
            output_dir: Output directory for YOLO format
        """
        logger.info("Converting dataset to YOLO format...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO directory structure
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Process each class directory
        train_count = 0
        val_count = 0
        
        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name not in self.all_classes:
                continue
            
            class_idx = self.class_to_idx[class_dir.name]
            image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            
            # Split into train/val (80/20)
            split_idx = int(len(image_files) * 0.8)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            # Process training files
            for img_file in train_files:
                self._process_yolo_image(
                    img_file, class_idx, output_dir / "images" / "train",
                    output_dir / "labels" / "train", f"train_{train_count:06d}"
                )
                train_count += 1
            
            # Process validation files
            for img_file in val_files:
                self._process_yolo_image(
                    img_file, class_idx, output_dir / "images" / "val",
                    output_dir / "labels" / "val", f"val_{val_count:06d}"
                )
                val_count += 1
        
        # Create dataset.yaml for YOLO
        dataset_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.all_classes),
            'names': self.all_classes
        }
        
        import yaml
        with open(output_dir / "dataset.yaml", 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        logger.info(f"YOLO dataset created: {train_count} train, {val_count} val images")
    
    def _process_yolo_image(self, img_file: Path, class_idx: int, 
                           img_output_dir: Path, label_output_dir: Path, 
                           new_name: str):
        """Process a single image for YOLO format."""
        # Copy image
        img_output_path = img_output_dir / f"{new_name}.jpg"
        shutil.copy2(img_file, img_output_path)
        
        # Create label file (assuming full image is the character)
        label_output_path = label_output_dir / f"{new_name}.txt"
        with open(label_output_path, 'w') as f:
            # YOLO format: class_id center_x center_y width height (normalized)
            f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
