#!/bin/bash
# Installation script for Jetson Nano Character Recognition System

set -e

echo "=========================================="
echo "Jetson Nano Character Recognition Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        print_status "Detected Jetson hardware"
        JETSON_VERSION=$(cat /etc/nv_tegra_release)
        print_status "Version: $JETSON_VERSION"
    else
        print_warning "Not running on Jetson hardware - some optimizations may not be available"
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Essential packages
    sudo apt-get install -y \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-serial-dev \
        libhdf5-dev \
        libhdf5-103 \
        python3-h5py
}

# Install OpenCV for Jetson
install_opencv_jetson() {
    print_status "Installing OpenCV for Jetson..."
    
    # Check if OpenCV is already installed
    if python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null; then
        print_status "OpenCV already installed"
        return
    fi
    
    # Install pre-built OpenCV for Jetson
    sudo apt-get install -y python3-opencv
    
    # Verify installation
    if python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null; then
        print_status "OpenCV installed successfully"
    else
        print_error "OpenCV installation failed"
        exit 1
    fi
}

# Install PyTorch for Jetson
install_pytorch_jetson() {
    print_status "Installing PyTorch for Jetson..."
    
    # Check if PyTorch is already installed
    if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
        print_status "PyTorch already installed"
        return
    fi
    
    # Install PyTorch wheel for Jetson
    # Note: This URL may need to be updated for different JetPack versions
    TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
    TORCH_FILE="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    
    print_status "Downloading PyTorch wheel..."
    wget -O $TORCH_FILE $TORCH_URL
    
    print_status "Installing PyTorch..."
    pip3 install $TORCH_FILE
    
    # Install torchvision
    sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
    pip3 install torchvision
    
    # Clean up
    rm -f $TORCH_FILE
    
    # Verify installation
    if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
        print_status "PyTorch installed successfully"
    else
        print_error "PyTorch installation failed"
        exit 1
    fi
}

# Setup swap file
setup_swap() {
    print_status "Setting up swap file..."
    
    # Check if swap already exists
    if swapon --show | grep -q "/swapfile"; then
        print_status "Swap file already exists"
        return
    fi
    
    # Create 4GB swap file
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    
    # Make permanent
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    
    print_status "Swap file created successfully"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip3 install --upgrade pip
    
    # Install requirements
    if [ -f requirements.txt ]; then
        pip3 install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing basic dependencies"
        pip3 install numpy opencv-python ultralytics albumentations PyYAML tqdm
    fi
}

# Install Jetson-specific packages
install_jetson_packages() {
    if [ -f /etc/nv_tegra_release ]; then
        print_status "Installing Jetson-specific packages..."
        
        # Install jetson-stats
        sudo -H pip3 install jetson-stats
        
        # Install jtop
        sudo systemctl enable jetson_stats.service
        sudo systemctl start jetson_stats.service
        
        print_status "Jetson packages installed"
    fi
}

# Setup performance optimizations
setup_performance() {
    if [ -f /etc/nv_tegra_release ]; then
        print_status "Setting up performance optimizations..."
        
        # Set maximum performance mode
        sudo nvpmodel -m 0
        
        # Maximize CPU clocks
        sudo jetson_clocks
        
        print_status "Performance optimizations applied"
    fi
}

# Install the package
install_package() {
    print_status "Installing Jetson Character Recognition package..."
    
    # Install in development mode
    pip3 install -e .
    
    print_status "Package installed successfully"
}

# Create desktop shortcut (optional)
create_desktop_shortcut() {
    print_status "Creating desktop shortcut..."
    
    DESKTOP_FILE="$HOME/Desktop/jetson-char-recognition.desktop"
    
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Jetson Character Recognition
Comment=Real-time character detection system
Exec=python3 -m scripts.run_detection
Icon=applications-science
Terminal=true
Categories=Science;Education;
EOF
    
    chmod +x "$DESKTOP_FILE"
    print_status "Desktop shortcut created"
}

# Run tests
run_tests() {
    print_status "Running system tests..."
    
    if python3 scripts/test_system.py; then
        print_status "All tests passed!"
    else
        print_warning "Some tests failed - check the output above"
    fi
}

# Main installation function
main() {
    print_status "Starting installation..."
    
    # Check system
    check_jetson
    
    # System setup
    update_system
    install_system_deps
    setup_swap
    
    # Install ML frameworks
    install_opencv_jetson
    install_pytorch_jetson
    
    # Install Python dependencies
    install_python_deps
    
    # Install Jetson-specific packages
    install_jetson_packages
    
    # Install the package
    install_package
    
    # Setup optimizations
    setup_performance
    
    # Optional components
    if [ "$1" = "--with-desktop" ]; then
        create_desktop_shortcut
    fi
    
    if [ "$1" = "--with-tests" ] || [ "$2" = "--with-tests" ]; then
        run_tests
    fi
    
    print_status "Installation completed successfully!"
    print_status ""
    print_status "Usage:"
    print_status "  Train model: python3 scripts/train_model.py"
    print_status "  Run detection: python3 scripts/run_detection.py <model_path>"
    print_status "  Test system: python3 scripts/test_system.py"
    print_status ""
    print_status "For more information, see README.md"
}

# Parse command line arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [--with-desktop] [--with-tests] [--help]"
        echo ""
        echo "Options:"
        echo "  --with-desktop    Create desktop shortcut"
        echo "  --with-tests      Run tests after installation"
        echo "  --help           Show this help message"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
