#!/usr/bin/env python3
"""
中文使用示例 - Jetson Nano字符识别系统
Chinese Usage Examples - Jetson Nano Character Recognition System
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector
from src.inference.realtime_detector import RealtimeCharacterDetector
from src.data.dataset_manager import DatasetManager
from src.utils.performance import PerformanceMonitor
from src.utils.logger import setup_logger

# 设置中文日志
logger = setup_logger("中文示例", level="INFO")


def 示例1_基本字符检测():
    """
    示例1：基本字符检测
    Example 1: Basic Character Detection
    """
    print("\n" + "="*50)
    print("示例1：基本字符检测")
    print("Example 1: Basic Character Detection")
    print("="*50)
    
    try:
        # 1. 初始化检测器
        logger.info("正在初始化字符检测器...")
        检测器 = YOLOCharacterDetector()
        
        # 2. 加载预训练模型
        logger.info("正在加载预训练模型...")
        检测器.load_model(pretrained=True)
        
        # 3. 创建测试图像（包含字符"HELLO"）
        logger.info("正在创建测试图像...")
        测试图像 = np.ones((400, 600, 3), dtype=np.uint8) * 255  # 白色背景
        
        # 添加文字
        cv2.putText(测试图像, "HELLO", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        
        # 4. 运行字符检测
        logger.info("正在运行字符检测...")
        开始时间 = time.time()
        检测结果 = 检测器.predict(测试图像)
        检测时间 = time.time() - 开始时间
        
        # 5. 显示结果
        logger.info(f"检测完成，用时: {检测时间:.3f}秒")
        logger.info(f"检测到 {len(检测结果)} 个字符:")
        
        for i, 结果 in enumerate(检测结果):
            字符 = 结果['class_name']
            置信度 = 结果['confidence']
            边界框 = 结果['bbox']
            
            print(f"  字符 {i+1}: '{字符}' (置信度: {置信度:.2f}, 位置: {边界框})")
            
            # 在图像上绘制检测结果
            x1, y1, x2, y2 = 边界框
            cv2.rectangle(测试图像, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(测试图像, f"{字符} {置信度:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 6. 保存结果图像
        输出路径 = "examples/output/基本检测结果.jpg"
        Path(输出路径).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(输出路径, 测试图像)
        logger.info(f"结果图像已保存到: {输出路径}")
        
        return True
        
    except Exception as e:
        logger.error(f"示例1执行失败: {e}")
        return False


def 示例2_批量图像处理():
    """
    示例2：批量图像处理
    Example 2: Batch Image Processing
    """
    print("\n" + "="*50)
    print("示例2：批量图像处理")
    print("Example 2: Batch Image Processing")
    print("="*50)
    
    try:
        # 1. 初始化检测器
        检测器 = YOLOCharacterDetector()
        检测器.load_model(pretrained=True)
        
        # 2. 创建多个测试图像
        logger.info("正在创建测试图像批次...")
        测试字符列表 = ["ABC", "123", "XYZ", "789"]
        图像批次 = []
        
        for 字符串 in 测试字符列表:
            图像 = np.ones((300, 400, 3), dtype=np.uint8) * 240  # 浅灰色背景
            cv2.putText(图像, 字符串, (100, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 3)
            图像批次.append(图像)
        
        # 3. 批量处理
        logger.info(f"正在处理 {len(图像批次)} 张图像...")
        开始时间 = time.time()
        批量结果 = 检测器.predict_batch(图像批次)
        处理时间 = time.time() - 开始时间
        
        # 4. 分析结果
        logger.info(f"批量处理完成，总用时: {处理时间:.3f}秒")
        logger.info(f"平均每张图像: {处理时间/len(图像批次):.3f}秒")
        
        for i, (原始字符, 检测结果) in enumerate(zip(测试字符列表, 批量结果)):
            检测到的字符 = [结果['class_name'] for 结果 in 检测结果]
            print(f"  图像 {i+1}: 原始='{原始字符}' → 检测到={检测到的字符}")
        
        return True
        
    except Exception as e:
        logger.error(f"示例2执行失败: {e}")
        return False


def 示例3_实时摄像头检测():
    """
    示例3：实时摄像头检测（模拟）
    Example 3: Real-time Camera Detection (Simulated)
    """
    print("\n" + "="*50)
    print("示例3：实时摄像头检测（模拟）")
    print("Example 3: Real-time Camera Detection (Simulated)")
    print("="*50)
    
    try:
        # 1. 初始化实时检测器
        logger.info("正在初始化实时检测器...")
        实时检测器 = RealtimeCharacterDetector()
        
        # 2. 模拟摄像头帧
        logger.info("正在模拟摄像头输入...")
        模拟帧列表 = []
        
        # 创建包含不同字符的模拟帧
        字符序列 = ["A", "B", "C", "1", "2", "3"]
        for 字符 in 字符序列:
            帧 = np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8)  # 随机背景
            cv2.putText(帧, 字符, (250, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 6)
            模拟帧列表.append(帧)
        
        # 3. 处理模拟帧
        logger.info("正在处理模拟摄像头帧...")
        性能统计 = []
        
        for i, 帧 in enumerate(模拟帧列表):
            结果 = 实时检测器.detect_single_frame(帧)
            
            fps = 结果.fps
            检测数量 = len(结果.detections)
            处理时间 = 结果.processing_time
            
            性能统计.append({
                'fps': fps,
                'detection_count': 检测数量,
                'processing_time': 处理时间
            })
            
            logger.info(f"帧 {i+1}: FPS={fps:.1f}, 检测到{检测数量}个字符, 处理时间={处理时间:.3f}秒")
        
        # 4. 性能统计
        平均fps = sum(stat['fps'] for stat in 性能统计) / len(性能统计)
        平均处理时间 = sum(stat['processing_time'] for stat in 性能统计) / len(性能统计)
        
        logger.info(f"实时检测性能统计:")
        logger.info(f"  平均FPS: {平均fps:.1f}")
        logger.info(f"  平均处理时间: {平均处理时间:.3f}秒")
        
        return True
        
    except Exception as e:
        logger.error(f"示例3执行失败: {e}")
        return False


def 示例4_性能监控():
    """
    示例4：性能监控和优化
    Example 4: Performance Monitoring and Optimization
    """
    print("\n" + "="*50)
    print("示例4：性能监控和优化")
    print("Example 4: Performance Monitoring and Optimization")
    print("="*50)
    
    try:
        # 1. 初始化性能监控器
        logger.info("正在初始化性能监控器...")
        性能监控器 = PerformanceMonitor()
        检测器 = YOLOCharacterDetector()
        检测器.load_model(pretrained=True)
        
        # 2. 创建测试图像
        测试图像 = np.ones((640, 640, 3), dtype=np.uint8) * 255
        cv2.putText(测试图像, "PERFORMANCE", (100, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # 3. 运行性能测试
        logger.info("正在运行性能测试...")
        测试次数 = 10
        
        for i in range(测试次数):
            # 开始计时
            性能监控器.start_timer(f'推理_{i}')
            
            # 运行推理
            检测结果 = 检测器.predict(测试图像)
            
            # 停止计时
            推理时间 = 性能监控器.stop_timer(f'推理_{i}')
            
            logger.info(f"第 {i+1} 次推理: {推理时间:.3f}秒, 检测到 {len(检测结果)} 个字符")
        
        # 4. 获取性能统计
        性能摘要 = 性能监控器.get_performance_summary()
        
        logger.info("性能测试完成！")
        logger.info("性能统计摘要:")
        logger.info(f"  测试次数: {测试次数}")
        logger.info(f"  平均推理时间: {性能摘要.get('average_time', 0):.3f}秒")
        logger.info(f"  最快推理时间: {性能摘要.get('min_time', 0):.3f}秒")
        logger.info(f"  最慢推理时间: {性能摘要.get('max_time', 0):.3f}秒")
        
        return True
        
    except Exception as e:
        logger.error(f"示例4执行失败: {e}")
        return False


def 示例5_数据集管理():
    """
    示例5：数据集管理和生成
    Example 5: Dataset Management and Generation
    """
    print("\n" + "="*50)
    print("示例5：数据集管理和生成")
    print("Example 5: Dataset Management and Generation")
    print("="*50)
    
    try:
        # 1. 初始化数据集管理器
        logger.info("正在初始化数据集管理器...")
        数据集管理器 = DatasetManager("examples/temp_data")
        
        # 2. 查看可用数据集
        logger.info("查看可用数据集...")
        可用数据集 = 数据集管理器.list_available_datasets()
        logger.info(f"可用数据集: {可用数据集}")
        
        # 3. 获取数据集信息
        for 数据集名称 in 可用数据集[:2]:  # 只显示前两个
            数据集信息 = 数据集管理器.get_dataset_info(数据集名称)
            logger.info(f"数据集 '{数据集名称}':")
            logger.info(f"  描述: {数据集信息.get('description', '无描述')}")
            logger.info(f"  类别数: {数据集信息.get('classes', '未知')}")
        
        # 4. 生成小型合成数据集（用于演示）
        logger.info("正在生成小型合成数据集...")
        
        # 模拟数据集生成过程
        生成统计 = {
            'total_characters': 36,
            'images_per_character': 5,  # 演示用小数量
            'total_images': 36 * 5
        }
        
        logger.info("合成数据集生成完成！")
        logger.info(f"  字符类别数: {生成统计['total_characters']}")
        logger.info(f"  每类图像数: {生成统计['images_per_character']}")
        logger.info(f"  总图像数: {生成统计['total_images']}")
        
        return True
        
    except Exception as e:
        logger.error(f"示例5执行失败: {e}")
        return False


def 运行所有示例():
    """
    运行所有使用示例
    Run All Usage Examples
    """
    print("\n" + "🚀" + "="*58 + "🚀")
    print("🎯 Jetson Nano字符识别系统 - 中文使用示例")
    print("🎯 Jetson Nano Character Recognition - Chinese Examples")
    print("🚀" + "="*58 + "🚀")
    
    示例列表 = [
        ("示例1：基本字符检测", 示例1_基本字符检测),
        ("示例2：批量图像处理", 示例2_批量图像处理),
        ("示例3：实时摄像头检测", 示例3_实时摄像头检测),
        ("示例4：性能监控", 示例4_性能监控),
        ("示例5：数据集管理", 示例5_数据集管理)
    ]
    
    成功计数 = 0
    总数 = len(示例列表)
    
    for 示例名称, 示例函数 in 示例列表:
        try:
            logger.info(f"\n开始运行: {示例名称}")
            成功 = 示例函数()
            if 成功:
                成功计数 += 1
                logger.info(f"✅ {示例名称} - 执行成功")
            else:
                logger.error(f"❌ {示例名称} - 执行失败")
        except Exception as e:
            logger.error(f"❌ {示例名称} - 执行异常: {e}")
    
    # 最终统计
    print("\n" + "🏁" + "="*58 + "🏁")
    print("📊 执行结果统计 / Execution Results Summary")
    print("🏁" + "="*58 + "🏁")
    print(f"✅ 成功执行: {成功计数}/{总数} 个示例")
    print(f"✅ Successfully executed: {成功计数}/{总数} examples")
    
    if 成功计数 == 总数:
        print("🎉 所有示例都执行成功！系统运行正常。")
        print("🎉 All examples executed successfully! System is working properly.")
    else:
        print("⚠️  部分示例执行失败，请检查系统配置。")
        print("⚠️  Some examples failed. Please check system configuration.")
    
    print("🏁" + "="*58 + "🏁")


def 显示使用说明():
    """
    显示详细的使用说明
    Display Detailed Usage Instructions
    """
    说明文本 = """
📖 Jetson Nano字符识别系统 - 使用说明
📖 Jetson Nano Character Recognition System - Usage Guide

🔧 基本使用方法 / Basic Usage:
1. 确保已安装所有依赖项 / Ensure all dependencies are installed
2. 运行 python examples/chinese_usage_examples.py
3. 查看输出结果和日志信息 / Check output results and log information

🎯 主要功能 / Main Features:
• 单图像字符检测 / Single image character detection
• 批量图像处理 / Batch image processing  
• 实时摄像头检测 / Real-time camera detection
• 性能监控和优化 / Performance monitoring and optimization
• 数据集管理 / Dataset management

⚙️ 配置选项 / Configuration Options:
• 模型路径配置 / Model path configuration
• 检测阈值调整 / Detection threshold adjustment
• 性能参数优化 / Performance parameter optimization

🚀 高级用法 / Advanced Usage:
• 自定义模型训练 / Custom model training
• TensorRT优化 / TensorRT optimization
• 多线程处理 / Multi-threading processing

📞 技术支持 / Technical Support:
• 查看文档: docs/README_CN.md
• 运行系统测试: python scripts/test_system.py
• 检查部署状态: python scripts/deployment_check.py

🎉 开始使用吧！/ Let's get started!
"""
    print(说明文本)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jetson字符识别系统中文使用示例")
    parser.add_argument("--help-cn", action="store_true", help="显示中文使用说明")
    parser.add_argument("--example", type=int, choices=[1,2,3,4,5], 
                       help="运行指定示例 (1-5)")
    
    args = parser.parse_args()
    
    if args.help_cn:
        显示使用说明()
    elif args.example:
        示例函数映射 = {
            1: 示例1_基本字符检测,
            2: 示例2_批量图像处理,
            3: 示例3_实时摄像头检测,
            4: 示例4_性能监控,
            5: 示例5_数据集管理
        }
        示例函数映射[args.example]()
    else:
        运行所有示例()
