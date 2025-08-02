#!/usr/bin/env python3
"""
Validate character detection on reference scene images.
验证参考场景图像上的字符检测。
"""

import cv2
import numpy as np
import argparse
import json
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector
from src.utils.logger import setup_logger
from src.utils.performance import PerformanceMonitor

logger = setup_logger("validate_reference", level="INFO")


class ReferenceSceneValidator:
    """Validator for reference scene character detection."""
    
    def __init__(self, model_path: str):
        """
        Initialize validator.
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.detector = YOLOCharacterDetector()
        self.performance_monitor = PerformanceMonitor()
        
        # Load model / 加载模型
        logger.info(f"Loading model: {model_path}")
        self.detector.load_model(model_path)
        
        # Expected detections for reference scenes / 参考场景的预期检测
        self.expected_detections = {
            'cylindrical_surface.jpg': {
                'expected_chars': ['5'],
                'min_confidence': 0.5,
                'description': 'Character "5" on cylindrical surface'
            },
            'geometric_background.jpg': {
                'expected_chars': ['M', 'M', 'M'],
                'min_confidence': 0.5,
                'description': 'Three "M" letters in hexagonal backgrounds'
            },
            'mixed_characters.jpg': {
                'expected_chars': ['A', 'B', 'C', '1', '2', '3', 'T', 'E', 'M', 'P', '2', '5', 'C', 'Z', 'O', 'N', 'E', 'A', '7', 'X', '9', 'K', '2'],
                'min_confidence': 0.3,  # Lower threshold for complex scene
                'description': 'Mixed characters in various contexts'
            }
        }
    
    def validate_single_image(self, image_path: Path) -> dict:
        """
        Validate character detection on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating image: {image_path.name}")
        
        # Load image / 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': f'Could not load image: {image_path}'}
        
        # Run detection / 运行检测
        self.performance_monitor.start_timer('detection')
        detections = self.detector.predict(image)
        detection_time = self.performance_monitor.stop_timer('detection')
        
        # Get expected results / 获取预期结果
        expected = self.expected_detections.get(image_path.name, {})
        expected_chars = expected.get('expected_chars', [])
        min_confidence = expected.get('min_confidence', 0.5)
        
        # Analyze detections / 分析检测结果
        detected_chars = [det['class_name'] for det in detections if det['confidence'] >= min_confidence]
        high_conf_detections = [det for det in detections if det['confidence'] >= min_confidence]
        
        # Calculate metrics / 计算指标
        true_positives = len([char for char in detected_chars if char in expected_chars])
        false_positives = len([char for char in detected_chars if char not in expected_chars])
        false_negatives = len([char for char in expected_chars if char not in detected_chars])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create annotated image / 创建标注图像
        annotated_image = self._create_annotated_image(image, high_conf_detections)
        
        # Compile results / 编译结果
        results = {
            'image_name': image_path.name,
            'description': expected.get('description', 'No description'),
            'detection_time': detection_time,
            'total_detections': len(detections),
            'high_confidence_detections': len(high_conf_detections),
            'detected_characters': detected_chars,
            'expected_characters': expected_chars,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'min_confidence_threshold': min_confidence,
            'detections': high_conf_detections,
            'annotated_image': annotated_image
        }
        
        # Log results / 记录结果
        logger.info(f"  Detected: {detected_chars}")
        logger.info(f"  Expected: {expected_chars}")
        logger.info(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
        logger.info(f"  Detection time: {detection_time:.3f}s")
        
        return results
    
    def _create_annotated_image(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Create annotated image with detection results."""
        annotated = image.copy()
        
        for detection in detections:
            # Extract detection info / 提取检测信息
            bbox = detection['bbox']
            char = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box / 绘制边界框
            color = (0, 255, 0) if confidence >= 0.7 else (0, 255, 255)  # Green for high conf, yellow for medium
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label / 绘制标签
            label = f"{char} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label / 标签背景
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Label text / 标签文本
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated
    
    def validate_all_reference_scenes(self, scenes_dir: Path, output_dir: Path = None) -> dict:
        """
        Validate all reference scene images.
        
        Args:
            scenes_dir: Directory containing reference scene images
            output_dir: Directory to save results and annotated images
            
        Returns:
            Complete validation results
        """
        logger.info(f"Validating reference scenes in: {scenes_dir}")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir = output_dir / 'annotated_images'
            annotated_dir.mkdir(exist_ok=True)
        
        # Find reference scene images / 找到参考场景图像
        image_extensions = ['.jpg', '.jpeg', '.png']
        scene_images = []
        
        for ext in image_extensions:
            scene_images.extend(scenes_dir.glob(f'*{ext}'))
        
        if not scene_images:
            logger.warning(f"No reference scene images found in: {scenes_dir}")
            return {'error': 'No images found'}
        
        # Validate each image / 验证每张图像
        all_results = {}
        overall_metrics = {
            'total_images': 0,
            'total_detections': 0,
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_false_negatives': 0,
            'total_detection_time': 0
        }
        
        for image_path in scene_images:
            if image_path.name in self.expected_detections:
                results = self.validate_single_image(image_path)
                
                if 'error' not in results:
                    all_results[image_path.name] = results
                    
                    # Update overall metrics / 更新总体指标
                    overall_metrics['total_images'] += 1
                    overall_metrics['total_detections'] += results['total_detections']
                    overall_metrics['total_true_positives'] += results['true_positives']
                    overall_metrics['total_false_positives'] += results['false_positives']
                    overall_metrics['total_false_negatives'] += results['false_negatives']
                    overall_metrics['total_detection_time'] += results['detection_time']
                    
                    # Save annotated image / 保存标注图像
                    if output_dir:
                        annotated_path = annotated_dir / f"annotated_{image_path.name}"
                        cv2.imwrite(str(annotated_path), results['annotated_image'])
                        logger.info(f"Saved annotated image: {annotated_path}")
        
        # Calculate overall metrics / 计算总体指标
        if overall_metrics['total_images'] > 0:
            overall_precision = overall_metrics['total_true_positives'] / (
                overall_metrics['total_true_positives'] + overall_metrics['total_false_positives']
            ) if (overall_metrics['total_true_positives'] + overall_metrics['total_false_positives']) > 0 else 0
            
            overall_recall = overall_metrics['total_true_positives'] / (
                overall_metrics['total_true_positives'] + overall_metrics['total_false_negatives']
            ) if (overall_metrics['total_true_positives'] + overall_metrics['total_false_negatives']) > 0 else 0
            
            overall_f1 = 2 * (overall_precision * overall_recall) / (
                overall_precision + overall_recall
            ) if (overall_precision + overall_recall) > 0 else 0
            
            avg_detection_time = overall_metrics['total_detection_time'] / overall_metrics['total_images']
            avg_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
            
            overall_metrics.update({
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1_score': overall_f1,
                'average_detection_time': avg_detection_time,
                'average_fps': avg_fps
            })
        
        # Compile complete results / 编译完整结果
        complete_results = {
            'model_path': self.model_path,
            'validation_timestamp': time.time(),
            'scenes_directory': str(scenes_dir),
            'overall_metrics': overall_metrics,
            'individual_results': all_results
        }
        
        # Save results / 保存结果
        if output_dir:
            results_file = output_dir / 'validation_results.json'
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._prepare_for_json(complete_results)
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Validation results saved to: {results_file}")
        
        # Log overall results / 记录总体结果
        logger.info("\n" + "="*50)
        logger.info("REFERENCE SCENE VALIDATION RESULTS")
        logger.info("="*50)
        logger.info(f"Images validated: {overall_metrics['total_images']}")
        logger.info(f"Overall precision: {overall_metrics.get('overall_precision', 0):.3f}")
        logger.info(f"Overall recall: {overall_metrics.get('overall_recall', 0):.3f}")
        logger.info(f"Overall F1 score: {overall_metrics.get('overall_f1_score', 0):.3f}")
        logger.info(f"Average detection time: {overall_metrics.get('average_detection_time', 0):.3f}s")
        logger.info(f"Average FPS: {overall_metrics.get('average_fps', 0):.1f}")
        logger.info("="*50)
        
        return complete_results
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by converting numpy arrays."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Validate character detection on reference scenes")
    parser.add_argument("model_path", help="Path to trained model file")
    parser.add_argument("--scenes-dir", default="data/samples/reference_scenes",
                       help="Directory containing reference scene images")
    parser.add_argument("--output-dir", default="validation_output",
                       help="Output directory for results and annotated images")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create reference sample images first")
    
    args = parser.parse_args()
    
    # Create reference samples if requested / 如果请求则创建参考样本
    if args.create_samples:
        logger.info("Creating reference sample images...")
        from data.samples.create_reference_samples import main as create_samples
        import sys
        
        # Temporarily modify sys.argv for create_samples / 临时修改sys.argv用于create_samples
        original_argv = sys.argv
        sys.argv = ['create_reference_samples.py', '--output-dir', args.scenes_dir]
        try:
            create_samples()
        finally:
            sys.argv = original_argv
    
    # Check if model exists / 检查模型是否存在
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Check if scenes directory exists / 检查场景目录是否存在
    scenes_dir = Path(args.scenes_dir)
    if not scenes_dir.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        logger.info("Use --create-samples to generate reference images")
        sys.exit(1)
    
    try:
        # Initialize validator / 初始化验证器
        validator = ReferenceSceneValidator(str(model_path))
        
        # Run validation / 运行验证
        output_dir = Path(args.output_dir)
        results = validator.validate_all_reference_scenes(scenes_dir, output_dir)
        
        # Print summary / 打印摘要
        if 'error' not in results:
            metrics = results['overall_metrics']
            print(f"\nValidation completed successfully!")
            print(f"Results saved to: {output_dir}")
            print(f"Overall F1 Score: {metrics.get('overall_f1_score', 0):.3f}")
            print(f"Average FPS: {metrics.get('average_fps', 0):.1f}")
        else:
            print(f"Validation failed: {results['error']}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
