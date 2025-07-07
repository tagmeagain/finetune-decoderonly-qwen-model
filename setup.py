#!/usr/bin/env python3
"""
Setup script for Qwen 1.5B fine-tuning project
This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"CUDA available: {cuda_version} ✓")
            print(f"GPU devices: {device_count}")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {device_name}")
            return True
        else:
            print("CUDA not available - will use CPU (training will be slow)")
            return False
    except ImportError:
        print("PyTorch not installed yet")
        return False

def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully ✓")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["models", "datasets", "logs"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting imports...")
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "pandas",
        "numpy",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "unsloth"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package} ✓")
        except ImportError as e:
            print(f"  {package} ✗")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        return False
    else:
        print("All imports successful ✓")
        return True

def main():
    """Main setup function."""
    print("Qwen 1.5B Fine-tuning Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return
    
    # Test imports
    if not test_imports():
        print("Some packages failed to import. Please check the installation.")
        return
    
    print("\n" + "=" * 40)
    print("Setup completed successfully! ✓")
    print("\nNext steps:")
    print("1. Prepare your dataset (use create_dataset.py or create your own CSV)")
    print("2. Run fine-tuning: python fine_tune_qwen.py")
    print("3. Test the model: python inference.py")
    
    if not cuda_available:
        print("\n⚠️  Warning: No GPU detected. Training will be very slow on CPU.")
        print("   Consider using a cloud service with GPU support.")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 