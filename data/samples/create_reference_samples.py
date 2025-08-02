#!/usr/bin/env python3
"""
Create reference sample images for testing character detection.
创建用于测试字符检测的参考样本图像。
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def create_cylindrical_surface_sample():
    """
    Create sample image with characters on cylindrical surface.
    创建圆柱表面字符的样本图像。
    """
    # Create image with sky background / 创建天空背景图像
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Sky gradient background / 天空渐变背景
    for y in range(600):
        intensity = int(255 * (1 - y / 600))
        color = (intensity, intensity, 255)  # Blue sky
        img[y, :] = color
    
    # Add some clouds / 添加云朵
    cloud_color = (240, 240, 255)
    cv2.ellipse(img, (200, 150), (80, 40), 0, 0, 360, cloud_color, -1)
    cv2.ellipse(img, (600, 100), (100, 50), 0, 0, 360, cloud_color, -1)
    cv2.ellipse(img, (700, 200), (60, 30), 0, 0, 360, cloud_color, -1)
    
    # Create cylindrical object / 创建圆柱形物体
    cylinder_center = (400, 350)
    cylinder_width = 120
    cylinder_height = 200
    
    # Cylinder body (white with shading) / 圆柱体（白色带阴影）
    cylinder_color = (245, 245, 245)
    cv2.rectangle(img, 
                  (cylinder_center[0] - cylinder_width//2, cylinder_center[1] - cylinder_height//2),
                  (cylinder_center[0] + cylinder_width//2, cylinder_center[1] + cylinder_height//2),
                  cylinder_color, -1)
    
    # Add cylindrical shading / 添加圆柱阴影
    for x in range(cylinder_center[0] - cylinder_width//2, cylinder_center[0] + cylinder_width//2):
        distance_from_center = abs(x - cylinder_center[0])
        shading_factor = distance_from_center / (cylinder_width // 2)
        shading = int(245 * (1 - 0.3 * shading_factor))
        
        cv2.line(img, 
                (x, cylinder_center[1] - cylinder_height//2),
                (x, cylinder_center[1] + cylinder_height//2),
                (shading, shading, shading), 1)
    
    # Add top and bottom ellipses / 添加顶部和底部椭圆
    cv2.ellipse(img, 
                (cylinder_center[0], cylinder_center[1] - cylinder_height//2),
                (cylinder_width//2, 15), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img, 
                (cylinder_center[0], cylinder_center[1] + cylinder_height//2),
                (cylinder_width//2, 15), 0, 0, 360, (200, 200, 200), -1)
    
    # Add character "5" on the cylinder / 在圆柱上添加字符"5"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 8
    text_color = (50, 50, 200)  # Dark blue
    
    # Calculate text position / 计算文本位置
    text = "5"
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = cylinder_center[0] - text_width // 2
    text_y = cylinder_center[1] + text_height // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Add some surface details / 添加表面细节
    # Horizontal lines to emphasize cylindrical shape / 水平线强调圆柱形状
    for y_offset in [-60, -20, 20, 60]:
        y_pos = cylinder_center[1] + y_offset
        cv2.line(img,
                (cylinder_center[0] - cylinder_width//2 + 10, y_pos),
                (cylinder_center[0] + cylinder_width//2 - 10, y_pos),
                (180, 180, 180), 1)
    
    return img


def create_geometric_background_sample():
    """
    Create sample image with characters on geometric background.
    创建几何背景字符的样本图像。
    """
    # Create base image / 创建基础图像
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # Gradient background / 渐变背景
    for y in range(500):
        for x in range(700):
            r = int(200 + 55 * np.sin(x * 0.01))
            g = int(220 + 35 * np.cos(y * 0.01))
            b = int(240 + 15 * np.sin((x + y) * 0.005))
            img[y, x] = (r, g, b)
    
    # Create hexagonal shapes with letters / 创建带字母的六边形
    hexagon_centers = [(150, 150), (350, 150), (550, 150)]
    hexagon_colors = [(180, 100, 200), (100, 180, 100), (200, 140, 100)]  # Purple, Green, Orange
    letters = ['M', 'M', 'M']
    
    for i, (center, color, letter) in enumerate(zip(hexagon_centers, hexagon_colors, letters)):
        # Create hexagon / 创建六边形
        hexagon_points = []
        radius = 60
        for angle in range(0, 360, 60):
            x = center[0] + int(radius * np.cos(np.radians(angle)))
            y = center[1] + int(radius * np.sin(np.radians(angle)))
            hexagon_points.append([x, y])
        
        hexagon_points = np.array(hexagon_points, np.int32)
        cv2.fillPoly(img, [hexagon_points], color)
        
        # Add border / 添加边框
        cv2.polylines(img, [hexagon_points], True, (255, 255, 255), 3)
        
        # Add letter / 添加字母
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5
        thickness = 6
        text_color = (255, 255, 255)  # White
        
        (text_width, text_height), baseline = cv2.getTextSize(letter, font, font_scale, thickness)
        text_x = center[0] - text_width // 2
        text_y = center[1] + text_height // 2
        
        cv2.putText(img, letter, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Add decorative elements / 添加装饰元素
    # Curved lines / 曲线
    for i in range(3):
        start_point = (50, 350 + i * 30)
        end_point = (650, 380 + i * 30)
        color = (255 - i * 50, 255 - i * 30, 255 - i * 20)
        
        # Create curved line using multiple line segments / 使用多个线段创建曲线
        points = []
        for x in range(start_point[0], end_point[0], 10):
            y = start_point[1] + int(20 * np.sin((x - start_point[0]) * 0.02))
            points.append((x, y))
        
        for j in range(len(points) - 1):
            cv2.line(img, points[j], points[j + 1], color, 3)
    
    return img


def create_mixed_character_scene():
    """
    Create sample image with mixed characters in various scenarios.
    创建包含各种场景中混合字符的样本图像。
    """
    # Create base image / 创建基础图像
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img.fill(240)  # Light gray background
    
    # License plate style / 车牌风格
    plate_rect = (50, 50, 300, 100)
    cv2.rectangle(img, (plate_rect[0], plate_rect[1]), 
                  (plate_rect[0] + plate_rect[2], plate_rect[1] + plate_rect[3]), 
                  (255, 255, 255), -1)
    cv2.rectangle(img, (plate_rect[0], plate_rect[1]), 
                  (plate_rect[0] + plate_rect[2], plate_rect[1] + plate_rect[3]), 
                  (0, 0, 0), 3)
    
    # Add license plate text / 添加车牌文本
    plate_text = "ABC123"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 3
    text_color = (0, 0, 0)
    
    (text_width, text_height), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    text_x = plate_rect[0] + (plate_rect[2] - text_width) // 2
    text_y = plate_rect[1] + (plate_rect[3] + text_height) // 2
    
    cv2.putText(img, plate_text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Digital display style / 数字显示风格
    display_rect = (450, 50, 300, 120)
    cv2.rectangle(img, (display_rect[0], display_rect[1]), 
                  (display_rect[0] + display_rect[2], display_rect[1] + display_rect[3]), 
                  (20, 20, 20), -1)
    cv2.rectangle(img, (display_rect[0], display_rect[1]), 
                  (display_rect[0] + display_rect[2], display_rect[1] + display_rect[3]), 
                  (100, 100, 100), 2)
    
    # Add digital display text / 添加数字显示文本
    display_text = "TEMP 25C"
    text_color = (0, 255, 0)  # Green
    font_scale = 1.5
    
    (text_width, text_height), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
    text_x = display_rect[0] + (display_rect[2] - text_width) // 2
    text_y = display_rect[1] + (display_rect[3] + text_height) // 2
    
    cv2.putText(img, display_text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Signage style / 标牌风格
    sign_rect = (200, 250, 400, 150)
    cv2.rectangle(img, (sign_rect[0], sign_rect[1]), 
                  (sign_rect[0] + sign_rect[2], sign_rect[1] + sign_rect[3]), 
                  (100, 150, 200), -1)
    cv2.rectangle(img, (sign_rect[0], sign_rect[1]), 
                  (sign_rect[0] + sign_rect[2], sign_rect[1] + sign_rect[3]), 
                  (255, 255, 255), 4)
    
    # Add signage text / 添加标牌文本
    sign_lines = ["ZONE", "A7"]
    text_color = (255, 255, 255)
    font_scale = 2.0
    
    for i, line in enumerate(sign_lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = sign_rect[0] + (sign_rect[2] - text_width) // 2
        text_y = sign_rect[1] + 60 + i * 60
        
        cv2.putText(img, line, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Add scattered individual characters / 添加散布的单个字符
    scattered_chars = [('X', (100, 500)), ('9', (300, 480)), ('K', (500, 520)), ('2', (650, 490))]
    
    for char, pos in scattered_chars:
        # Add background circle / 添加背景圆圈
        cv2.circle(img, pos, 30, (200, 200, 255), -1)
        cv2.circle(img, pos, 30, (0, 0, 0), 2)
        
        # Add character / 添加字符
        (text_width, text_height), _ = cv2.getTextSize(char, font, 1.5, 3)
        text_x = pos[0] - text_width // 2
        text_y = pos[1] + text_height // 2
        
        cv2.putText(img, char, (text_x, text_y), font, 1.5, (0, 0, 0), 3)
    
    return img


def create_challenging_samples():
    """
    Create challenging test cases for character detection.
    创建字符检测的挑战性测试用例。
    """
    samples = {}
    
    # Low contrast sample / 低对比度样本
    low_contrast = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.putText(low_contrast, "LOW", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (180, 180, 180), 5)
    samples['low_contrast'] = low_contrast
    
    # Rotated characters / 旋转字符
    rotated = np.zeros((400, 600, 3), dtype=np.uint8)
    rotated.fill(255)
    
    # Create rotated text / 创建旋转文本
    center = (300, 200)
    angle = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "ROTATED"
    
    # Get text size / 获取文本大小
    (text_width, text_height), _ = cv2.getTextSize(text, font, 2, 3)
    
    # Create rotation matrix / 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Create temporary image for text / 为文本创建临时图像
    temp_img = np.zeros((400, 600, 3), dtype=np.uint8)
    temp_img.fill(255)
    cv2.putText(temp_img, text, (center[0] - text_width//2, center[1] + text_height//2), 
                font, 2, (0, 0, 0), 3)
    
    # Apply rotation / 应用旋转
    rotated = cv2.warpAffine(temp_img, rotation_matrix, (600, 400))
    samples['rotated'] = rotated
    
    # Partially occluded / 部分遮挡
    occluded = np.ones((400, 600, 3), dtype=np.uint8) * 240
    cv2.putText(occluded, "HIDDEN", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)
    
    # Add occlusion rectangles / 添加遮挡矩形
    cv2.rectangle(occluded, (200, 150), (280, 220), (100, 100, 100), -1)
    cv2.rectangle(occluded, (350, 160), (420, 210), (150, 150, 150), -1)
    samples['occluded'] = occluded
    
    return samples


def main():
    """Main function to create all reference samples."""
    parser = argparse.ArgumentParser(description="Create reference sample images")
    parser.add_argument("--output-dir", default="data/samples/reference_scenes",
                       help="Output directory for sample images")
    
    args = parser.parse_args()
    
    # Create output directory / 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating reference sample images...")
    
    # Create cylindrical surface sample / 创建圆柱表面样本
    print("Creating cylindrical surface sample...")
    cylindrical_img = create_cylindrical_surface_sample()
    cv2.imwrite(str(output_dir / "cylindrical_surface.jpg"), cylindrical_img)
    
    # Create geometric background sample / 创建几何背景样本
    print("Creating geometric background sample...")
    geometric_img = create_geometric_background_sample()
    cv2.imwrite(str(output_dir / "geometric_background.jpg"), geometric_img)
    
    # Create mixed character scene / 创建混合字符场景
    print("Creating mixed character scene...")
    mixed_img = create_mixed_character_scene()
    cv2.imwrite(str(output_dir / "mixed_characters.jpg"), mixed_img)
    
    # Create challenging samples / 创建挑战性样本
    print("Creating challenging samples...")
    challenging_dir = output_dir / "challenging"
    challenging_dir.mkdir(exist_ok=True)
    
    challenging_samples = create_challenging_samples()
    for name, img in challenging_samples.items():
        cv2.imwrite(str(challenging_dir / f"{name}.jpg"), img)
    
    print(f"Reference samples created in: {output_dir}")
    print("Sample files:")
    print("  - cylindrical_surface.jpg")
    print("  - geometric_background.jpg") 
    print("  - mixed_characters.jpg")
    print("  - challenging/low_contrast.jpg")
    print("  - challenging/rotated.jpg")
    print("  - challenging/occluded.jpg")


if __name__ == "__main__":
    main()
