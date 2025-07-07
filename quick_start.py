#!/usr/bin/env python3
"""
Quick Start script for Qwen 1.5B fine-tuning
This script guides you through the entire process from setup to fine-tuning.
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_file_exists(filename):
    """Check if a file exists."""
    return os.path.exists(filename)

def interactive_choice(prompt, options):
    """Get user choice from a list of options."""
    while True:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        try:
            choice = int(input(f"\nEnter your choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return choice
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")

def main():
    """Main quick start function."""
    print("🚀 Qwen 1.5B Fine-tuning Quick Start")
    print("=" * 50)
    print("This script will guide you through the entire fine-tuning process.")
    print()
    
    # Step 1: Setup
    print("Step 1: Environment Setup")
    print("-" * 30)
    
    setup_choice = interactive_choice(
        "Do you want to run the setup script?",
        ["Yes, run setup script", "Skip setup (already done)", "Exit"]
    )
    
    if setup_choice == 3:
        print("Goodbye!")
        return
    elif setup_choice == 1:
        if not run_command("python setup.py", "Running setup script"):
            print("Setup failed. Please run 'python setup.py' manually and try again.")
            return
    
    # Step 2: Dataset preparation
    print("\nStep 2: Dataset Preparation")
    print("-" * 30)
    
    if check_file_exists("sample_dataset.csv"):
        print("✓ Found sample_dataset.csv")
        dataset_choice = interactive_choice(
            "Dataset options:",
            ["Use existing sample_dataset.csv", "Create new dataset", "Use custom dataset file"]
        )
    else:
        print("No sample dataset found.")
        dataset_choice = interactive_choice(
            "Dataset options:",
            ["Create new dataset", "Use custom dataset file", "Exit"]
        )
        dataset_choice += 1  # Adjust for missing option
    
    if dataset_choice == 3:
        print("Goodbye!")
        return
    elif dataset_choice == 1:
        print("Using existing sample_dataset.csv")
    elif dataset_choice == 2:
        if not run_command("python create_dataset.py", "Creating new dataset"):
            print("Dataset creation failed. Please run 'python create_dataset.py' manually.")
            return
    
    # Step 3: Configuration
    print("\nStep 3: Training Configuration")
    print("-" * 30)
    
    print("Current configuration:")
    print("- Model: Qwen/Qwen1.5-1.8B")
    print("- Epochs: 3")
    print("- Batch size: 2")
    print("- Learning rate: 2e-4")
    print("- Max sequence length: 2048")
    
    config_choice = interactive_choice(
        "Configuration options:",
        ["Use default configuration", "Modify configuration", "Continue with defaults"]
    )
    
    if config_choice == 2:
        print("\nTo modify configuration, edit the variables in fine_tune_qwen.py:")
        print("- MODEL_NAME: Change the model")
        print("- NUM_EPOCHS: Number of training epochs")
        print("- BATCH_SIZE: Training batch size")
        print("- LEARNING_RATE: Learning rate")
        print("- MAX_SEQ_LENGTH: Maximum sequence length")
        
        input("\nPress Enter after making changes...")
    
    # Step 4: Fine-tuning
    print("\nStep 4: Start Fine-tuning")
    print("-" * 30)
    
    print("⚠️  Important Notes:")
    print("- Fine-tuning requires significant computational resources")
    print("- GPU is highly recommended (training on CPU will be very slow)")
    print("- The process may take several hours depending on your hardware")
    print("- Make sure you have enough disk space (at least 10GB free)")
    
    start_training = interactive_choice(
        "Ready to start fine-tuning?",
        ["Yes, start training", "Not yet, I need to prepare", "Exit"]
    )
    
    if start_training == 3:
        print("Goodbye!")
        return
    elif start_training == 2:
        print("\nPreparation checklist:")
        print("□ Ensure you have a GPU (recommended)")
        print("□ Check available disk space (10GB+)")
        print("□ Verify your dataset is ready")
        print("□ Close other resource-intensive applications")
        print("\nRun this script again when ready.")
        return
    
    # Start training
    print("\n🚀 Starting fine-tuning...")
    print("This may take several hours. You can monitor progress in the output.")
    print("To stop training, press Ctrl+C")
    print("\n" + "="*50)
    
    if not run_command("python fine_tune_qwen.py", "Running fine-tuning"):
        print("Fine-tuning failed. Check the error messages above.")
        return
    
    # Step 5: Testing
    print("\nStep 5: Test Your Fine-tuned Model")
    print("-" * 30)
    
    if check_file_exists("qwen-finetuned"):
        print("✓ Fine-tuned model found!")
        
        test_choice = interactive_choice(
            "Test options:",
            ["Run inference script", "Skip testing", "Exit"]
        )
        
        if test_choice == 1:
            print("\n🧪 Testing your fine-tuned model...")
            run_command("python inference.py", "Running inference")
        elif test_choice == 3:
            print("Goodbye!")
            return
    else:
        print("Fine-tuned model not found. Training may have failed.")
    
    # Final message
    print("\n" + "="*50)
    print("🎉 Quick Start Complete!")
    print("\nWhat's next:")
    print("1. Your fine-tuned model is saved in 'qwen-finetuned/' directory")
    print("2. Use 'python inference.py' to test your model anytime")
    print("3. Experiment with different datasets and configurations")
    print("4. Check the README.md for advanced usage and troubleshooting")
    
    print("\nHappy fine-tuning! 🚀")

if __name__ == "__main__":
    main() 