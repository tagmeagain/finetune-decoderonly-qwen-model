#!/usr/bin/env python3
"""
Specialized installation script for Unsloth dependencies
This script handles the PyTorch 2.7+ requirement for Unsloth.
"""

import subprocess
import sys
import platform
import os

def get_pytorch_command():
    """Get the appropriate PyTorch installation command based on system."""
    system = platform.system().lower()
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA detected: {cuda_version}")
            return f"pip install torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    except ImportError:
        pass
    
    # Default to CPU version
    return "pip install torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

def install_pytorch():
    """Install PyTorch 2.7+ first."""
    print("Installing PyTorch 2.7+ (required for Unsloth)...")
    
    pytorch_cmd = get_pytorch_command()
    print(f"Running: {pytorch_cmd}")
    
    try:
        subprocess.check_call(pytorch_cmd, shell=True)
        print("✓ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing PyTorch: {e}")
        return False

def install_unsloth():
    """Install Unsloth after PyTorch."""
    print("\nInstalling Unsloth...")
    
    try:
        # Install Unsloth
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git"
        ])
        print("✓ Unsloth installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing Unsloth: {e}")
        return False

def install_other_dependencies():
    """Install other required dependencies."""
    print("\nInstalling other dependencies...")
    
    dependencies = [
        "transformers>=4.36.0",
        "datasets>=2.14.0", 
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "accelerate>=0.20.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0"
    ]
    
    try:
        for dep in dependencies:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        print("✓ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify that all packages are installed correctly."""
    print("\nVerifying installation...")
    
    packages_to_check = [
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
    
    for package in packages_to_check:
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
        print("✓ All packages imported successfully")
        return True

def main():
    """Main installation function."""
    print("🔧 Unsloth Dependencies Installation")
    print("=" * 40)
    print("This script will install PyTorch 2.7+ and Unsloth with all dependencies.")
    print()
    
    # Step 1: Install PyTorch
    if not install_pytorch():
        print("Failed to install PyTorch. Please install manually:")
        print("pip install torch>=2.7.0 torchvision torchaudio")
        return
    
    # Step 2: Install Unsloth
    if not install_unsloth():
        print("Failed to install Unsloth. Please check the error messages above.")
        return
    
    # Step 3: Install other dependencies
    if not install_other_dependencies():
        print("Failed to install other dependencies. Please install manually:")
        print("pip install transformers datasets pandas numpy accelerate peft trl bitsandbytes scipy scikit-learn tqdm wandb")
        return
    
    # Step 4: Verify installation
    if not verify_installation():
        print("Some packages failed to import. Please check the installation.")
        return
    
    print("\n" + "=" * 40)
    print("🎉 Installation completed successfully!")
    print("\nYou can now run:")
    print("  python quick_start.py")
    print("  python fine_tune_qwen.py")
    print("  python inference.py")

if __name__ == "__main__":
    main() 