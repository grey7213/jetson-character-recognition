#!/usr/bin/env python3
"""
Training script for character recognition model.
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector
from src.data.dataset_manager import DatasetManager
from src.utils.logger import setup_logger
from src.utils.config_loader import config_loader

logger = setup_logger("train_model", level="INFO")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Character Recognition Model")
    parser.add_argument("--dataset", default="synthetic", 
                       choices=["synthetic", "emnist", "chars74k"],
                       help="Dataset to use for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--output-dir", default="models",
                       help="Output directory for trained models")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory")
    parser.add_argument("--resume", type=str,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    logger.info("Starting model training")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    try:
        # Initialize dataset manager
        dataset_manager = DatasetManager(args.data_dir)
        
        # Download/prepare dataset
        logger.info(f"Preparing dataset: {args.dataset}")
        dataset_dir = dataset_manager.download_dataset(args.dataset)
        
        # Convert to YOLO format
        yolo_dataset_dir = Path(args.data_dir) / f"{args.dataset}_yolo"
        dataset_manager.prepare_yolo_dataset(dataset_dir, yolo_dataset_dir)
        
        # Initialize model
        detector = YOLOCharacterDetector()
        
        # Load model (pretrained or resume)
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            detector.load_model(args.resume)
        else:
            logger.info("Loading pretrained model")
            detector.load_model(pretrained=True)
        
        # Start training
        dataset_yaml = yolo_dataset_dir / "dataset.yaml"
        results = detector.train(
            dataset_path=str(dataset_yaml),
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save trained model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        model_path = output_dir / f"character_detector_{timestamp}.pt"
        detector.save_model(str(model_path))
        
        logger.info(f"Training completed successfully")
        logger.info(f"Model saved to: {model_path}")
        
        # Run validation
        logger.info("Running validation")
        metrics = detector.validate(str(dataset_yaml))
        logger.info(f"Validation metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
