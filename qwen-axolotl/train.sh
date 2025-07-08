#!/bin/bash

# Axolotl training script for Qwen 2.1.5
echo "Starting Axolotl training for Qwen 2.1.5..."

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "No GPU detected, will use CPU"
fi

# Create necessary directories
mkdir -p outputs logs lora_outputs

# Start training with Axolotl
echo "Starting training..."
axolotl train train.yml

echo "Training completed!"
echo "Check outputs/ directory for the trained model"
echo "Check logs/ directory for training logs" 