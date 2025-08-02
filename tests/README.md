# Tests Directory / 测试目录

## Overview / 概述

This directory contains comprehensive test suites for the Jetson Character Recognition system, including unit tests, integration tests, and performance benchmarks.

本目录包含Jetson字符识别系统的综合测试套件，包括单元测试、集成测试和性能基准测试。

## Directory Structure / 目录结构

```
tests/
├── README.md                    # This file / 本文件
├── conftest.py                  # Pytest configuration / Pytest配置
├── requirements.txt             # Test dependencies / 测试依赖
├── unit/                        # Unit tests / 单元测试
│   ├── __init__.py
│   ├── test_models/            # Model tests / 模型测试
│   │   ├── test_yolo_detector.py
│   │   ├── test_tensorrt_optimizer.py
│   │   └── __init__.py
│   ├── test_data/              # Data processing tests / 数据处理测试
│   │   ├── test_dataset_manager.py
│   │   ├── test_augmentation.py
│   │   └── __init__.py
│   ├── test_inference/         # Inference tests / 推理测试
│   │   ├── test_realtime_detector.py
│   │   ├── test_camera_handler.py
│   │   └── __init__.py
│   └── test_utils/             # Utility tests / 工具测试
│       ├── test_config_loader.py
│       ├── test_jetson_utils.py
│       ├── test_performance.py
│       └── __init__.py
├── integration/                 # Integration tests / 集成测试
│   ├── __init__.py
│   ├── test_end_to_end.py      # End-to-end pipeline tests / 端到端流水线测试
│   ├── test_training_pipeline.py # Training pipeline tests / 训练流水线测试
│   ├── test_inference_pipeline.py # Inference pipeline tests / 推理流水线测试
│   └── test_data_pipeline.py   # Data pipeline tests / 数据流水线测试
├── performance/                 # Performance tests / 性能测试
│   ├── __init__.py
│   ├── test_speed_benchmarks.py # Speed benchmarks / 速度基准测试
│   ├── test_memory_usage.py    # Memory usage tests / 内存使用测试
│   ├── test_accuracy_benchmarks.py # Accuracy benchmarks / 精度基准测试
│   └── test_jetson_performance.py # Jetson-specific performance / Jetson特定性能测试
├── fixtures/                    # Test fixtures and data / 测试固件和数据
│   ├── sample_images/          # Sample test images / 测试样本图像
│   ├── mock_models/            # Mock model files / 模拟模型文件
│   ├── test_configs/           # Test configuration files / 测试配置文件
│   └── expected_outputs/       # Expected test outputs / 预期测试输出
└── reports/                     # Test reports and coverage / 测试报告和覆盖率
    ├── coverage/               # Coverage reports / 覆盖率报告
    ├── performance/            # Performance test results / 性能测试结果
    └── integration/            # Integration test results / 集成测试结果
```

## Test Categories / 测试类别

### 1. Unit Tests / 单元测试

Test individual components in isolation.

独立测试各个组件。

**Coverage Areas / 覆盖范围:**
- Model loading and inference / 模型加载和推理
- Data processing and augmentation / 数据处理和增强
- Configuration management / 配置管理
- Utility functions / 工具函数
- Error handling / 错误处理

**Running Unit Tests / 运行单元测试:**
```bash
# Run all unit tests / 运行所有单元测试
pytest tests/unit/ -v

# Run specific test module / 运行特定测试模块
pytest tests/unit/test_models/test_yolo_detector.py -v

# Run with coverage / 运行并生成覆盖率报告
pytest tests/unit/ --cov=src --cov-report=html
```

### 2. Integration Tests / 集成测试

Test component interactions and complete workflows.

测试组件交互和完整工作流程。

**Test Scenarios / 测试场景:**
- End-to-end detection pipeline / 端到端检测流水线
- Training workflow / 训练工作流程
- Data processing pipeline / 数据处理流水线
- Model optimization pipeline / 模型优化流水线

**Running Integration Tests / 运行集成测试:**
```bash
# Run all integration tests / 运行所有集成测试
pytest tests/integration/ -v

# Run specific integration test / 运行特定集成测试
pytest tests/integration/test_end_to_end.py -v
```

### 3. Performance Tests / 性能测试

Benchmark system performance and resource usage.

基准测试系统性能和资源使用。

**Metrics Tested / 测试指标:**
- Inference speed (FPS) / 推理速度 (FPS)
- Memory usage / 内存使用
- Model accuracy / 模型精度
- Power consumption / 功耗
- Latency / 延迟

**Running Performance Tests / 运行性能测试:**
```bash
# Run all performance tests / 运行所有性能测试
pytest tests/performance/ -v

# Run speed benchmarks / 运行速度基准测试
pytest tests/performance/test_speed_benchmarks.py -v

# Generate performance report / 生成性能报告
pytest tests/performance/ --benchmark-json=reports/performance/benchmark.json
```

## Test Configuration / 测试配置

### Pytest Configuration / Pytest配置

Located in `conftest.py`:

位于 `conftest.py`：

- Test fixtures / 测试固件
- Mock objects / 模拟对象
- Test data setup / 测试数据设置
- Custom markers / 自定义标记

### Test Dependencies / 测试依赖

Located in `requirements.txt`:

位于 `requirements.txt`：

```
pytest>=6.0.0
pytest-cov>=2.12.0
pytest-benchmark>=3.4.0
pytest-mock>=3.6.0
pytest-xdist>=2.3.0
```

## Test Data / 测试数据

### Sample Images / 样本图像

Located in `fixtures/sample_images/`:

位于 `fixtures/sample_images/`：

- Single character images / 单字符图像
- Multi-character scenes / 多字符场景
- Edge cases and challenging scenarios / 边缘情况和挑战性场景
- Reference images for validation / 验证参考图像

### Mock Models / 模拟模型

Located in `fixtures/mock_models/`:

位于 `fixtures/mock_models/`：

- Lightweight test models / 轻量级测试模型
- Model metadata files / 模型元数据文件
- Configuration files / 配置文件

## Running Tests / 运行测试

### Complete Test Suite / 完整测试套件

```bash
# Run all tests / 运行所有测试
pytest tests/ -v

# Run with coverage report / 运行并生成覆盖率报告
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run tests in parallel / 并行运行测试
pytest tests/ -n auto
```

### Specific Test Categories / 特定测试类别

```bash
# Unit tests only / 仅单元测试
pytest tests/unit/ -v

# Integration tests only / 仅集成测试
pytest tests/integration/ -v

# Performance tests only / 仅性能测试
pytest tests/performance/ -v
```

### Test Filtering / 测试过滤

```bash
# Run tests by marker / 按标记运行测试
pytest -m "not slow" tests/

# Run tests by keyword / 按关键字运行测试
pytest -k "test_model" tests/

# Run specific test function / 运行特定测试函数
pytest tests/unit/test_models/test_yolo_detector.py::test_model_loading -v
```

## Test Markers / 测试标记

Custom pytest markers for test categorization:

用于测试分类的自定义pytest标记：

- `@pytest.mark.unit` - Unit tests / 单元测试
- `@pytest.mark.integration` - Integration tests / 集成测试
- `@pytest.mark.performance` - Performance tests / 性能测试
- `@pytest.mark.slow` - Slow running tests / 运行缓慢的测试
- `@pytest.mark.jetson` - Jetson-specific tests / Jetson特定测试
- `@pytest.mark.gpu` - GPU-required tests / 需要GPU的测试

## Coverage Requirements / 覆盖率要求

**Target Coverage / 目标覆盖率:**
- Overall: >80% / 总体：>80%
- Core modules: >90% / 核心模块：>90%
- Critical functions: 100% / 关键函数：100%

**Coverage Exclusions / 覆盖率排除:**
- Test files / 测试文件
- Configuration files / 配置文件
- Third-party integrations / 第三方集成
- Platform-specific code (when not available) / 平台特定代码（不可用时）

## Continuous Integration / 持续集成

### GitHub Actions / GitHub Actions

Test automation configuration:

测试自动化配置：

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Test Reports / 测试报告

### Coverage Reports / 覆盖率报告

Generated in `reports/coverage/`:

生成在 `reports/coverage/`：

- HTML coverage report / HTML覆盖率报告
- XML coverage data / XML覆盖率数据
- Terminal coverage summary / 终端覆盖率摘要

### Performance Reports / 性能报告

Generated in `reports/performance/`:

生成在 `reports/performance/`：

- Benchmark results / 基准测试结果
- Performance trends / 性能趋势
- Resource usage statistics / 资源使用统计

## Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **Test failures due to missing dependencies / 因缺少依赖导致测试失败**
   ```bash
   pip install -r tests/requirements.txt
   ```

2. **GPU tests failing on CPU-only systems / GPU测试在仅CPU系统上失败**
   ```bash
   pytest tests/ -m "not gpu"
   ```

3. **Slow test execution / 测试执行缓慢**
   ```bash
   pytest tests/ -m "not slow" --maxfail=1
   ```

### Test Debugging / 测试调试

```bash
# Run with verbose output / 详细输出运行
pytest tests/ -v -s

# Stop on first failure / 首次失败时停止
pytest tests/ --maxfail=1

# Run specific failing test / 运行特定失败测试
pytest tests/unit/test_models/test_yolo_detector.py::test_model_loading -v -s
```

## Contributing to Tests / 贡献测试

### Writing New Tests / 编写新测试

1. Follow naming convention: `test_*.py` / 遵循命名约定
2. Use descriptive test names / 使用描述性测试名称
3. Include docstrings / 包含文档字符串
4. Add appropriate markers / 添加适当标记
5. Ensure test isolation / 确保测试隔离

### Test Review Checklist / 测试审查清单

- [ ] Test covers edge cases / 测试覆盖边缘情况
- [ ] Test is deterministic / 测试是确定性的
- [ ] Test has clear assertions / 测试有明确断言
- [ ] Test documentation is complete / 测试文档完整
- [ ] Test runs in reasonable time / 测试在合理时间内运行
