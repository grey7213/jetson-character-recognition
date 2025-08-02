"""
TensorRT optimization for Jetson Nano deployment.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TensorRTOptimizer:
    """TensorRT optimization utilities for Jetson Nano."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize TensorRT optimizer.
        
        Args:
            config: TensorRT configuration
        """
        self.config = config or {}
        self.max_workspace_size = self.config.get('max_workspace_size', 1 << 30)  # 1GB
        self.max_batch_size = self.config.get('max_batch_size', 1)
        self.fp16_mode = self.config.get('fp16', True)
        self.int8_mode = self.config.get('int8', False)
        
        # Check if TensorRT is available
        self.tensorrt_available = self._check_tensorrt_availability()
        
    def _check_tensorrt_availability(self) -> bool:
        """Check if TensorRT is available on the system."""
        try:
            import tensorrt as trt
            logger.info(f"TensorRT version: {trt.__version__}")
            return True
        except ImportError:
            logger.warning("TensorRT not available. Optimization will be limited.")
            return False
    
    def optimize_onnx_to_tensorrt(self, 
                                  onnx_path: str, 
                                  output_path: str,
                                  input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> str:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path for TensorRT engine output
            input_shape: Input tensor shape (batch, channels, height, width)
            
        Returns:
            Path to generated TensorRT engine
        """
        if not self.tensorrt_available:
            raise RuntimeError("TensorRT is not available")
        
        try:
            import tensorrt as trt
            
            logger.info(f"Converting ONNX model to TensorRT: {onnx_path}")
            
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("ONNX parsing failed")
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            
            # Enable FP16 precision if requested
            if self.fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 optimization enabled")
            
            # Enable INT8 precision if requested
            if self.int8_mode and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 optimization enabled")
                # Note: INT8 calibration would be needed for production use
            
            # Set optimization profiles
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            
            # Set dynamic shapes (min, opt, max)
            min_shape = input_shape
            opt_shape = input_shape
            max_shape = (input_shape[0] * 2, input_shape[1], input_shape[2], input_shape[3])
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("Building TensorRT engine...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            raise
    
    def optimize_pytorch_to_tensorrt(self, 
                                     model: torch.nn.Module,
                                     output_path: str,
                                     input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                                     example_input: Optional[torch.Tensor] = None) -> str:
        """
        Convert PyTorch model to TensorRT using torch2trt.
        
        Args:
            model: PyTorch model
            output_path: Path for TensorRT model output
            input_shape: Input tensor shape
            example_input: Example input tensor
            
        Returns:
            Path to optimized model
        """
        try:
            from torch2trt import torch2trt
            
            logger.info("Converting PyTorch model to TensorRT using torch2trt")
            
            # Prepare example input
            if example_input is None:
                example_input = torch.randn(input_shape).cuda()
            
            # Convert model
            model_trt = torch2trt(
                model,
                [example_input],
                fp16_mode=self.fp16_mode,
                int8_mode=self.int8_mode,
                max_workspace_size=self.max_workspace_size,
                max_batch_size=self.max_batch_size
            )
            
            # Save optimized model
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_trt.state_dict(), output_path)
            
            logger.info(f"TensorRT optimized model saved to: {output_path}")
            return output_path
            
        except ImportError:
            logger.error("torch2trt not available. Please install torch2trt for PyTorch optimization.")
            raise
        except Exception as e:
            logger.error(f"PyTorch to TensorRT optimization failed: {e}")
            raise
    
    def benchmark_model(self, 
                        model_path: str,
                        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                        num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark TensorRT model performance.
        
        Args:
            model_path: Path to TensorRT engine
            input_shape: Input tensor shape
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics
        """
        if not self.tensorrt_available:
            raise RuntimeError("TensorRT is not available")
        
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger.info(f"Benchmarking TensorRT model: {model_path}")
            
            # Load engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(model_path, 'rb') as f:
                engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate buffers
            input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
            output_size = 1000 * np.dtype(np.float32).itemsize  # Estimate
            
            h_input = cuda.pagelocked_empty(np.prod(input_shape), dtype=np.float32)
            h_output = cuda.pagelocked_empty(1000, dtype=np.float32)
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            # Create CUDA stream
            stream = cuda.Stream()
            
            # Warm up
            for _ in range(10):
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
            
            # Benchmark
            import time
            times = []
            
            for _ in range(num_iterations):
                start_time = time.time()
                
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate metrics
            times = np.array(times)
            metrics = {
                'mean_inference_time': float(np.mean(times)),
                'std_inference_time': float(np.std(times)),
                'min_inference_time': float(np.min(times)),
                'max_inference_time': float(np.max(times)),
                'fps': float(1.0 / np.mean(times)),
                'throughput': float(num_iterations / np.sum(times))
            }
            
            logger.info(f"Benchmark results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}
    
    def create_calibration_dataset(self, 
                                   images_dir: str,
                                   output_path: str,
                                   num_samples: int = 100) -> str:
        """
        Create calibration dataset for INT8 quantization.
        
        Args:
            images_dir: Directory containing calibration images
            output_path: Path to save calibration data
            num_samples: Number of calibration samples
            
        Returns:
            Path to calibration dataset
        """
        logger.info(f"Creating calibration dataset from {images_dir}")
        
        try:
            import cv2
            
            images_path = Path(images_dir)
            image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
            
            if len(image_files) < num_samples:
                logger.warning(f"Only {len(image_files)} images found, using all available")
                num_samples = len(image_files)
            
            # Select random subset
            np.random.shuffle(image_files)
            selected_files = image_files[:num_samples]
            
            # Process images
            calibration_data = []
            for img_file in selected_files:
                img = cv2.imread(str(img_file))
                img = cv2.resize(img, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                calibration_data.append(img)
            
            # Save calibration data
            calibration_array = np.array(calibration_data)
            np.save(output_path, calibration_array)
            
            logger.info(f"Calibration dataset saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create calibration dataset: {e}")
            raise
    
    def get_optimization_recommendations(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimization recommendations based on model characteristics.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'tensorrt': True,
            'fp16': True,
            'int8': False,
            'dynamic_shapes': False,
            'workspace_size': '1GB'
        }
        
        # Adjust based on model size
        if 'parameters' in model_info:
            param_count = model_info['parameters']
            if param_count > 50_000_000:  # Large model
                recommendations['int8'] = True
                recommendations['workspace_size'] = '2GB'
            elif param_count < 1_000_000:  # Small model
                recommendations['fp16'] = False  # May not provide significant benefit
        
        # Adjust based on input size
        if 'input_size' in model_info:
            input_size = model_info['input_size']
            if input_size[0] > 640 or input_size[1] > 640:
                recommendations['dynamic_shapes'] = True
                recommendations['workspace_size'] = '2GB'
        
        return recommendations
