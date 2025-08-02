#!/usr/bin/env python3
"""
Synthetic character data generator for training.
合成字符数据生成器，用于训练。
"""

import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import random
import string

class SyntheticDataGenerator:
    """Generate synthetic character images for training."""
    
    def __init__(self, output_dir: str = "data/raw/synthetic"):
        """
        Initialize the synthetic data generator.
        
        Args:
            output_dir: Output directory for generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Character classes
        self.digits = [str(i) for i in range(10)]
        self.letters = [chr(ord('A') + i) for i in range(26)]
        self.all_characters = self.digits + self.letters
        
        # Font configurations
        self.fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX
        ]
        
        # Generation parameters
        self.image_size = (64, 64)
        self.variations_per_character = 100
        
        print(f"Initialized SyntheticDataGenerator")
        print(f"Output directory: {self.output_dir}")
        print(f"Characters: {len(self.all_characters)} classes")
    
    def generate_character_image(self, character: str, variation: int = 0) -> np.ndarray:
        """
        Generate a single character image with variations.
        
        Args:
            character: Character to generate
            variation: Variation number for different styles
            
        Returns:
            Generated image as numpy array
        """
        # Create blank image
        img = np.zeros(self.image_size, dtype=np.uint8)
        
        # Font variations
        font = self.fonts[variation % len(self.fonts)]
        font_scale = 1.5 + (variation % 3) * 0.3  # 1.5 to 2.4
        thickness = 2 + (variation % 2)  # 2 or 3
        
        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(
            character, font, font_scale, thickness
        )
        
        # Calculate position (with slight random offset)
        x = (self.image_size[1] - text_width) // 2 + random.randint(-5, 5)
        y = (self.image_size[0] + text_height) // 2 + random.randint(-5, 5)
        
        # Ensure text stays within image bounds
        x = max(0, min(x, self.image_size[1] - text_width))
        y = max(text_height, min(y, self.image_size[0]))
        
        # Draw the character
        cv2.putText(img, character, (x, y), font, font_scale, 255, thickness)
        
        # Apply augmentations
        img = self._apply_augmentations(img, variation)
        
        return img
    
    def _apply_augmentations(self, img: np.ndarray, variation: int) -> np.ndarray:
        """Apply various augmentations to the image."""
        
        # Add Gaussian noise (25% chance)
        if variation % 4 == 0:
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # Add rotation (20% chance)
        if variation % 5 == 0:
            angle = (variation % 21) - 10  # -10 to +10 degrees
            center = (img.shape[1] // 2, img.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, img.shape[::-1])
        
        # Add blur (15% chance)
        if variation % 7 == 0:
            kernel_size = 3 if variation % 2 == 0 else 5
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Adjust brightness and contrast (30% chance)
        if variation % 3 == 0:
            alpha = 0.8 + (variation % 5) * 0.1  # 0.8 to 1.2
            beta = -20 + (variation % 5) * 10    # -20 to 20
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        return img
    
    def generate_dataset(self, count_per_character: int = None) -> Dict[str, Any]:
        """
        Generate complete synthetic dataset.
        
        Args:
            count_per_character: Number of images per character
            
        Returns:
            Dataset generation statistics
        """
        if count_per_character is None:
            count_per_character = self.variations_per_character
        
        print(f"Generating {count_per_character} images per character...")
        
        stats = {
            'total_characters': len(self.all_characters),
            'images_per_character': count_per_character,
            'total_images': len(self.all_characters) * count_per_character,
            'generated_files': [],
            'class_distribution': {}
        }
        
        for char_idx, character in enumerate(self.all_characters):
            print(f"Generating images for '{character}' ({char_idx + 1}/{len(self.all_characters)})")
            
            # Create character directory
            char_dir = self.output_dir / character
            char_dir.mkdir(exist_ok=True)
            
            char_files = []
            
            for variation in range(count_per_character):
                # Generate image
                img = self.generate_character_image(character, variation)
                
                # Save image
                filename = f"{character}_{variation:03d}.png"
                filepath = char_dir / filename
                cv2.imwrite(str(filepath), img)
                
                char_files.append(str(filepath))
            
            stats['generated_files'].extend(char_files)
            stats['class_distribution'][character] = len(char_files)
        
        # Save generation statistics
        stats_file = self.output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset generation completed!")
        print(f"Total images generated: {stats['total_images']}")
        print(f"Statistics saved to: {stats_file}")
        
        return stats
    
    def generate_yolo_annotations(self) -> str:
        """
        Generate YOLO format annotations for the synthetic dataset.
        
        Returns:
            Path to the generated dataset.yaml file
        """
        print("Generating YOLO format annotations...")
        
        # Create YOLO format directories
        yolo_dir = self.output_dir.parent / "processed" / "yolo_format"
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        
        for dir_path in [yolo_dir, images_dir, labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        train_files = []
        val_files = []
        
        for char_idx, character in enumerate(self.all_characters):
            char_dir = self.output_dir / character
            
            if not char_dir.exists():
                continue
            
            image_files = list(char_dir.glob("*.png"))
            
            # Split into train/val (80/20)
            split_idx = int(len(image_files) * 0.8)
            train_images = image_files[:split_idx]
            val_images = image_files[split_idx:]
            
            # Process training images
            for img_file in train_images:
                # Copy image
                new_img_path = images_dir / f"{character}_{img_file.stem}.png"
                img = cv2.imread(str(img_file))
                cv2.imwrite(str(new_img_path), img)
                
                # Create label file (single character centered)
                label_file = labels_dir / f"{character}_{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    # YOLO format: class_id center_x center_y width height
                    f.write(f"{char_idx} 0.5 0.5 0.8 0.8\n")
                
                train_files.append(str(new_img_path))
            
            # Process validation images
            for img_file in val_images:
                # Copy image
                new_img_path = images_dir / f"{character}_{img_file.stem}.png"
                img = cv2.imread(str(img_file))
                cv2.imwrite(str(new_img_path), img)
                
                # Create label file
                label_file = labels_dir / f"{character}_{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write(f"{char_idx} 0.5 0.5 0.8 0.8\n")
                
                val_files.append(str(new_img_path))
        
        # Create train.txt and val.txt
        with open(yolo_dir / "train.txt", 'w') as f:
            f.write('\n'.join(train_files))
        
        with open(yolo_dir / "val.txt", 'w') as f:
            f.write('\n'.join(val_files))
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(yolo_dir.absolute()),
            'train': 'train.txt',
            'val': 'val.txt',
            'nc': len(self.all_characters),
            'names': self.all_characters
        }
        
        dataset_yaml = yolo_dir / "dataset.yaml"
        with open(dataset_yaml, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"YOLO annotations generated!")
        print(f"Training images: {len(train_files)}")
        print(f"Validation images: {len(val_files)}")
        print(f"Dataset config: {dataset_yaml}")
        
        return str(dataset_yaml)
    
    def create_sample_images(self):
        """Create sample images for testing and demonstration."""
        print("Creating sample images...")
        
        samples_dir = self.output_dir.parent / "samples"
        single_chars_dir = samples_dir / "single_characters"
        multi_chars_dir = samples_dir / "multi_characters"
        
        for dir_path in [samples_dir, single_chars_dir, multi_chars_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create single character samples
        for character in self.all_characters:
            img = self.generate_character_image(character, 0)
            # Resize for better visibility
            img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            
            if character.isdigit():
                filename = f"digit_{character}.png"
            else:
                filename = f"letter_{character}.png"
            
            cv2.imwrite(str(single_chars_dir / filename), img_resized)
        
        # Create multi-character samples
        self._create_multi_character_samples(multi_chars_dir)
        
        print(f"Sample images created in: {samples_dir}")
    
    def _create_multi_character_samples(self, output_dir: Path):
        """Create multi-character sample images."""
        
        # License plate style
        license_img = np.zeros((100, 300, 3), dtype=np.uint8)
        license_img.fill(255)  # White background
        
        license_text = "ABC123"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        color = (0, 0, 0)  # Black text
        
        (text_width, text_height), _ = cv2.getTextSize(license_text, font, font_scale, thickness)
        x = (license_img.shape[1] - text_width) // 2
        y = (license_img.shape[0] + text_height) // 2
        
        cv2.putText(license_img, license_text, (x, y), font, font_scale, color, thickness)
        cv2.imwrite(str(output_dir / "license_plate.jpg"), license_img)
        
        # Digital display style
        display_img = np.zeros((150, 400, 3), dtype=np.uint8)
        display_img.fill(20)  # Dark background
        
        display_text = "TEMP 25C"
        color = (0, 255, 0)  # Green text
        
        (text_width, text_height), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
        x = (display_img.shape[1] - text_width) // 2
        y = (display_img.shape[0] + text_height) // 2
        
        cv2.putText(display_img, display_text, (x, y), font, font_scale, color, thickness)
        cv2.imwrite(str(output_dir / "display_panel.jpg"), display_img)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Generate synthetic character dataset")
    parser.add_argument("--output", default="data/raw/synthetic", 
                       help="Output directory for generated data")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of images per character")
    parser.add_argument("--yolo", action="store_true",
                       help="Generate YOLO format annotations")
    parser.add_argument("--samples", action="store_true",
                       help="Create sample images")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SyntheticDataGenerator(args.output)
    
    # Generate dataset
    stats = generator.generate_dataset(args.count)
    
    # Generate YOLO annotations if requested
    if args.yolo:
        generator.generate_yolo_annotations()
    
    # Create sample images if requested
    if args.samples:
        generator.create_sample_images()
    
    print("\nGeneration completed successfully!")
    print(f"Total images: {stats['total_images']}")
    print(f"Output directory: {generator.output_dir}")


if __name__ == "__main__":
    main()
