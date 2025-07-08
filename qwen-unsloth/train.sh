#!/bin/bash

# Unsloth training script for Qwen
echo "🚀 Starting Qwen fine-tuning with Unsloth..."

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "⚠️  No GPU detected, will use CPU (this will be very slow)"
fi

# Check if dataset exists
if [ ! -f "alpaca_dataset.json" ]; then
    echo "❌ Dataset file 'alpaca_dataset.json' not found!"
    echo "Please create the dataset file before running training."
    exit 1
fi

# Create necessary directories
mkdir -p outputs logs

# Start training
echo "🎯 Starting training..."
python fine_tune_qwen.py

echo "✅ Training completed!"
echo "📁 Check outputs/ directory for the trained model"
echo "📊 Check logs/ directory for training logs" 