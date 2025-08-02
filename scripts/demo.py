#!/usr/bin/env python3
"""
Demo script for Jetson Character Recognition System.
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector
from src.inference.realtime_detector import RealtimeCharacterDetector
from src.data.dataset_manager import DatasetManager
from src.utils.logger import setup_logger
from src.utils.jetson_utils import jetson_monitor

logger = setup_logger("demo", level="INFO")


class CharacterRecognitionDemo:
    """Interactive demo for character recognition system."""
    
    def __init__(self):
        """Initialize demo."""
        self.detector = None
        self.realtime_detector = None
        
    def run_interactive_demo(self):
        """Run interactive demo menu."""
        while True:
            self._print_menu()
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                self.demo_system_info()
            elif choice == '2':
                self.demo_dataset_generation()
            elif choice == '3':
                self.demo_model_training()
            elif choice == '4':
                self.demo_single_image()
            elif choice == '5':
                self.demo_camera_test()
            elif choice == '6':
                self.demo_realtime_detection()
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
    
    def _print_menu(self):
        """Print demo menu."""
        print("\n" + "="*50)
        print("JETSON CHARACTER RECOGNITION DEMO")
        print("="*50)
        print("1. System Information")
        print("2. Generate Synthetic Dataset")
        print("3. Train Model (Quick Demo)")
        print("4. Single Image Detection")
        print("5. Camera Test")
        print("6. Real-time Detection")
        print("7. Exit")
        print("="*50)
    
    def demo_system_info(self):
        """Demo system information."""
        print("\n--- System Information ---")
        
        # Get system info
        system_info = jetson_monitor.get_system_info()
        
        print(f"Platform: {'Jetson' if system_info['is_jetson'] else 'Generic Linux'}")
        print(f"CPU Cores: {system_info['cpu_count']}")
        print(f"Total Memory: {system_info['memory_total'] / (1024**3):.1f} GB")
        print(f"Available Memory: {system_info['memory_available'] / (1024**3):.1f} GB")
        
        if system_info['is_jetson']:
            print(f"Jetson Model: {system_info.get('model', 'Unknown')}")
            print(f"CUDA Version: {system_info.get('cuda_version', 'Unknown')}")
        
        # Performance metrics
        print("\n--- Current Performance ---")
        metrics = jetson_monitor.get_performance_metrics()
        print(f"CPU Usage: {metrics['cpu_percent']:.1f}%")
        print(f"Memory Usage: {metrics['memory_percent']:.1f}%")
        
        if metrics['temperature']:
            print(f"Temperature: {metrics['temperature']:.1f}°C")
        
        if metrics['power_consumption']:
            print(f"Power: {metrics['power_consumption']:.1f}W")
    
    def demo_dataset_generation(self):
        """Demo synthetic dataset generation."""
        print("\n--- Synthetic Dataset Generation ---")
        
        try:
            dataset_manager = DatasetManager("demo_data")
            
            print("Generating synthetic character dataset...")
            dataset_dir = dataset_manager.download_dataset("synthetic")
            
            print(f"Dataset created at: {dataset_dir}")
            
            # Show some statistics
            class_dirs = list(dataset_dir.iterdir())
            print(f"Number of character classes: {len(class_dirs)}")
            
            # Show sample from first class
            if class_dirs:
                first_class = class_dirs[0]
                images = list(first_class.glob("*.png"))
                print(f"Sample class '{first_class.name}': {len(images)} images")
                
                if images:
                    # Display first image
                    img = cv2.imread(str(images[0]))
                    if img is not None:
                        cv2.imshow(f"Sample: {first_class.name}", img)
                        cv2.waitKey(2000)  # Show for 2 seconds
                        cv2.destroyAllWindows()
            
            print("Synthetic dataset generation completed!")
            
        except Exception as e:
            print(f"Error generating dataset: {e}")
    
    def demo_model_training(self):
        """Demo model training (quick version)."""
        print("\n--- Model Training Demo ---")
        print("Note: This is a quick demo with minimal training")
        
        try:
            # Initialize detector
            detector = YOLOCharacterDetector()
            
            print("Loading pretrained model...")
            detector.load_model(pretrained=True)
            
            print("Model loaded successfully!")
            
            # Show model info
            model_info = detector.get_model_info()
            print(f"Model type: {model_info.get('model_type', 'Unknown')}")
            print(f"Number of classes: {model_info.get('num_classes', 'Unknown')}")
            print(f"Input size: {model_info.get('input_size', 'Unknown')}")
            
            # Note: Actual training would require dataset preparation
            print("\nFor full training, use: python3 scripts/train_model.py")
            
        except Exception as e:
            print(f"Error in model training demo: {e}")
    
    def demo_single_image(self):
        """Demo single image detection."""
        print("\n--- Single Image Detection ---")
        
        # Create test image
        test_image = self._create_demo_image()
        
        try:
            # Initialize detector
            if self.detector is None:
                self.detector = YOLOCharacterDetector()
                print("Loading model...")
                self.detector.load_model(pretrained=True)
            
            print("Running detection on test image...")
            detections = self.detector.predict(test_image)
            
            print(f"Found {len(detections)} detections:")
            for i, detection in enumerate(detections):
                print(f"  {i+1}. Character: {detection['class_name']}, "
                      f"Confidence: {detection['confidence']:.2f}")
            
            # Draw results
            result_image = test_image.copy()
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                cv2.putText(result_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display result
            cv2.imshow("Detection Result", result_image)
            print("Press any key to close the image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in single image detection: {e}")
    
    def demo_camera_test(self):
        """Demo camera testing."""
        print("\n--- Camera Test ---")
        
        try:
            from src.inference.camera_handler import CameraHandler
            
            camera = CameraHandler()
            
            print("Testing camera...")
            results = camera.test_camera(duration=5.0)
            
            print(f"Test status: {results['status']}")
            if results['status'] == 'success':
                print(f"Success rate: {results['success_rate']:.1f}%")
                print(f"Actual FPS: {results['actual_fps']:.1f}")
                print(f"Total frames: {results['total_frames']}")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Camera test failed: {e}")
            print("This is normal if no camera is connected.")
    
    def demo_realtime_detection(self):
        """Demo real-time detection."""
        print("\n--- Real-time Detection Demo ---")
        print("Note: This requires a connected camera")
        
        try:
            if self.realtime_detector is None:
                print("Initializing real-time detector...")
                self.realtime_detector = RealtimeCharacterDetector()
            
            print("Starting real-time detection...")
            print("Press 'q' in the preview window to stop")
            
            # Set up detection callback
            def detection_callback(result):
                if result.detections:
                    characters = [d['class_name'] for d in result.detections]
                    print(f"Detected: {', '.join(characters)}")
            
            self.realtime_detector.set_detection_callback(detection_callback)
            
            # Start detection
            if self.realtime_detector.start_detection():
                print("Real-time detection started successfully!")
                print("Detection is running... Press Ctrl+C to stop")
                
                try:
                    import time
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping detection...")
                    self.realtime_detector.stop_detection()
            else:
                print("Failed to start real-time detection")
                print("Make sure a camera is connected")
            
        except Exception as e:
            print(f"Real-time detection demo failed: {e}")
    
    def _create_demo_image(self):
        """Create a demo image with characters."""
        # Create white background
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(img, "CHARACTER RECOGNITION DEMO", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add sample characters
        characters = ["A", "B", "C", "1", "2", "3", "X", "Y", "Z"]
        positions = [(100, 150), (200, 150), (300, 150), 
                    (100, 250), (200, 250), (300, 250),
                    (100, 350), (200, 350), (300, 350)]
        
        for char, pos in zip(characters, positions):
            cv2.putText(img, char, pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Jetson Character Recognition Demo")
    parser.add_argument("--mode", choices=["interactive", "auto"], default="interactive",
                       help="Demo mode")
    
    args = parser.parse_args()
    
    print("Welcome to the Jetson Character Recognition Demo!")
    print("This demo showcases the capabilities of the system.")
    
    demo = CharacterRecognitionDemo()
    
    if args.mode == "interactive":
        demo.run_interactive_demo()
    else:
        # Auto mode - run all demos
        print("\nRunning automatic demo...")
        demo.demo_system_info()
        demo.demo_dataset_generation()
        demo.demo_single_image()
        demo.demo_camera_test()
        print("\nDemo completed!")


if __name__ == "__main__":
    main()
