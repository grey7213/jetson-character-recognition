# Jetson Nano Character Recognition System

## Project Overview
A computer vision system for autonomous boat navigation that detects and recognizes alphanumeric characters (0-9, A-Z) from camera input, optimized for Jetson Nano hardware.

## Features
- Real-time character detection and recognition
- Support for digits (0-9) and letters (A-Z)
- Optimized for Jetson Nano hardware constraints
- Camera input processing
- Deployable package for client systems

## System Requirements
- **Hardware**: NVIDIA Jetson Nano
- **Software**: Python 3.6+, OpenCV, PyTorch/TensorFlow
- **Camera**: USB/CSI camera compatible with Jetson Nano

## Project Structure
```
jetson_character_recognition/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                 # Installation script
├── config/                  # Configuration files
│   ├── model_config.yaml   # Model configuration
│   └── camera_config.yaml  # Camera settings
├── src/                     # Source code
│   ├── __init__.py
│   ├── models/             # Model definitions
│   ├── data/               # Data processing
│   ├── utils/              # Utility functions
│   └── inference/          # Inference pipeline
├── models/                  # Trained model files
├── data/                    # Dataset and samples
├── scripts/                 # Training and deployment scripts
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd jetson_character_recognition

# Install on Jetson Nano
chmod +x scripts/install_jetson.sh
./scripts/install_jetson.sh --with-tests

# Or install manually
pip install -r requirements.txt
pip install -e .
```

### Usage
```bash
# Run interactive demo
python scripts/demo.py

# Train a model
python scripts/train_model.py --dataset synthetic --epochs 50

# Run real-time detection
python scripts/run_detection.py models/character_detector.pt

# Test the system
python scripts/test_system.py
```

## Performance Targets
- **Inference Speed**: >10 FPS on Jetson Nano
- **Accuracy**: >95% for clear characters
- **Memory Usage**: <2GB RAM
- **Power Consumption**: <10W

## Documentation
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Complete deployment instructions
- [API Documentation](docs/API.md) - Code API reference
- [Performance Tuning](docs/PERFORMANCE.md) - Optimization guidelines

## Examples

### Basic Usage
```python
from src.models.yolo_character_detector import YOLOCharacterDetector
from src.inference.realtime_detector import RealtimeCharacterDetector

# Load model and run detection
detector = YOLOCharacterDetector()
detector.load_model("models/character_detector.pt")

# Single image detection
import cv2
image = cv2.imread("test_image.jpg")
detections = detector.predict(image)

# Real-time detection
realtime_detector = RealtimeCharacterDetector("models/character_detector.pt")
realtime_detector.start_detection()
```

### Training Custom Model
```python
from src.data.dataset_manager import DatasetManager
from src.models.yolo_character_detector import YOLOCharacterDetector

# Prepare dataset
dataset_manager = DatasetManager()
dataset_dir = dataset_manager.download_dataset("synthetic")
dataset_manager.prepare_yolo_dataset(dataset_dir, "yolo_dataset")

# Train model
detector = YOLOCharacterDetector()
detector.load_model(pretrained=True)
results = detector.train("yolo_dataset/dataset.yaml", epochs=100)
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support
For issues and questions:
- Check the [troubleshooting guide](docs/DEPLOYMENT_GUIDE.md#troubleshooting)
- Run system diagnostics: `python scripts/test_system.py`
- Create an issue on GitHub

## License
MIT License
