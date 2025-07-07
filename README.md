# Qwen 1.5B Fine-tuning with Unsloth

This project provides a complete setup for fine-tuning the Qwen 1.5B model using Unsloth for efficient training. The setup includes data preparation, model fine-tuning, and inference capabilities.

## Features

- **Docker Support**: Complete containerized setup to avoid environment issues
- **LoRA Fine-tuning**: Uses Unsloth with LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning
- **Multiple LoRA Configurations**: Pre-configured settings for different hardware and requirements
- **4-bit Quantization**: Memory-efficient training with Unsloth optimizations
- **CSV Dataset Support**: Loads data from CSV files with System/User/Assistant format
- **Easy Configuration**: Simple configuration for training parameters
- **Inference Script**: Test your fine-tuned model with interactive prompts
- **Sample Dataset**: Includes a sample machine learning Q&A dataset
- **Hardware Analysis**: Automatic hardware detection and configuration recommendations
- **No Wandb Dependencies**: Clean setup without external logging dependencies

## Project Structure

```
fine-tune-qwen/
├── sample_dataset.csv            # Sample dataset for testing
├── fine_tune_qwen.py             # Basic fine-tuning script
├── fine_tune_qwen_lora.py        # Enhanced LoRA fine-tuning script (distributed ready)
├── lora_config.py                # LoRA configuration presets
├── choose_lora_config.py         # Interactive LoRA config chooser
├── inference.py                  # Inference script for testing
├── create_dataset.py             # Dataset creation utility
├── Dockerfile                    # Docker container definition
├── DOCKER_COMMANDS.md            # Simple Docker usage guide
└── README.md                     # This file
```

## Quick Start with Docker

### 1. Build the Docker image
```bash
docker build -t qwen-finetune .
```

### 2. Run the container
```bash
docker run -d --name qwen-container --gpus all -p 8888:8888 qwen-finetune
```

### 3. Get inside the container
```bash
docker exec -it qwen-container bash
```

### 4. Run distributed training
```bash
# For 2 GPUs
torchrun --nproc_per_node=2 --master_port=29500 fine_tune_lora.py

# For 4 GPUs  
torchrun --nproc_per_node=4 --master_port=29500 fine_tune_lora.py

# For 8 GPUs
torchrun --nproc_per_node=8 --master_port=29500 fine_tune_lora.py
```

See `DOCKER_COMMANDS.md` for complete Docker usage guide.

## Dataset Format

Your CSV dataset should have the following columns:
- `System`: System prompts (can be empty)
- `user`: User questions/queries
- `assistant`: Assistant responses/answers

Example:
```csv
System,user,assistant
,"What is machine learning?","Machine learning is a subset of artificial intelligence..."
,"How does a neural network work?","A neural network is a computational model..."
```

## Usage

### 1. Prepare Your Dataset

Replace `sample_dataset.csv` with your own dataset, or modify the script to point to your CSV file.

### 2. Choose LoRA Configuration (Optional)

```bash
python choose_lora_config.py
```

This will analyze your hardware and recommend the best LoRA configuration.

### 3. Fine-tune the Model

**Docker Commands:**
```bash
# Single GPU training
torchrun --nproc_per_node=1 --master_port=29500 fine_tune_qwen_lora.py

# Multi-GPU training (all available GPUs)
torchrun --nproc_per_node=auto --master_port=29500 fine_tune_qwen_lora.py
```

**Configuration Options** (modify in `fine_tune_qwen_lora.py`):
- `MODEL_NAME`: Model to fine-tune (default: "Qwen/Qwen1.5-1.8B")
- `DATASET_PATH`: Path to your CSV dataset
- `LORA_CONFIG_NAME`: LoRA configuration (default: "balanced")
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `BATCH_SIZE`: Training batch size (default: 2)

### 4. Test the Fine-tuned Model

```bash
python inference.py
```

This will:
- Load your fine-tuned model
- Run through sample questions
- Start an interactive mode where you can ask your own questions

## LoRA Configurations

The project includes multiple pre-configured LoRA settings optimized for different hardware and requirements:

### Available Configurations

1. **Minimal** (`r=4`): For very limited resources (<8GB GPU)
2. **Efficient** (`r=8`): Memory-efficient for limited resources (8-12GB GPU)
3. **Balanced** (`r=16`): Good balance of performance and efficiency (12-24GB GPU)
4. **Performance** (`r=32`): Higher performance with more parameters (24GB+ GPU)
5. **Custom**: Fully customizable configuration

### LoRA Parameters Explained

- **Rank (r)**: Controls the complexity of LoRA adaptation (4-64, higher = more parameters)
- **Alpha**: Scaling factor for LoRA weights (usually set to r or 2*r)
- **Dropout**: Regularization to prevent overfitting (0.0-0.2)
- **Target Modules**: Which layers to apply LoRA to (attention, MLP, or both)
- **RSLoRA**: Rank Stabilized LoRA for better stability

### Training Optimizations

- **4-bit Quantization**: Reduces memory usage significantly
- **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning
- **Gradient Checkpointing**: Saves memory during training
- **Mixed Precision Training**: Uses FP16 for faster training

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB (for CPU training)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space
- **CUDA**: Version 12.1 or higher

### Multi-GPU Setup
- **GPUs**: 2-8 NVIDIA GPUs for distributed training
- **NCCL**: Automatically configured in Docker
- **Memory**: 8GB+ VRAM per GPU recommended

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in the script
   - Use a lower LoRA rank configuration
   - Enable gradient checkpointing

2. **PyTorch Version Issues**
   - Docker setup includes correct PyTorch 2.7+ version
   - All dependencies are pre-installed in container

3. **Dataset Loading Issues**
   - Ensure CSV format matches the expected columns
   - Check file permissions and paths

4. **Distributed Training Issues**
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check NCCL environment variables
   - Use different master ports for multiple runs

### Performance Tips

1. **Memory Optimization**
   - Use 4-bit quantization (enabled by default)
   - Enable gradient checkpointing
   - Use appropriate LoRA rank for your GPU

2. **Speed Optimization**
   - Use mixed precision training (enabled by default)
   - Increase batch size if memory allows
   - Use multiple GPUs for distributed training

3. **Quality Optimization**
   - Use higher LoRA rank for better quality
   - Increase training epochs
   - Use better quality training data

## License

This project is for educational and research purposes. Please ensure you comply with the licenses of the models and datasets you use.

## Contributing

Feel free to submit issues and enhancement requests! 