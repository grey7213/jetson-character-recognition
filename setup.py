"""
Setup script for Jetson Character Recognition System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="jetson-character-recognition",
    version="1.0.0",
    author="Jetson Character Recognition Team",
    author_email="support@example.com",
    description="Computer vision system for character recognition on Jetson Nano",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/jetson-character-recognition",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "jetson": [
            "jetson-stats",
            "jtop",
        ],
    },
    # Note: Console scripts can be run directly from scripts/ directory
    # entry_points={
    #     "console_scripts": [
    #         "jetson-char-detect=scripts.run_detection:main",
    #         "jetson-char-train=scripts.train_model:main",
    #         "jetson-char-test=scripts.test_system:main",
    #     ],
    # },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md", "*.txt"],
    },
    zip_safe=False,
)
