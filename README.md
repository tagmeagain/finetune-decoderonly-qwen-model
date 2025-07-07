# Qwen 1.5B Fine-tuning with Unsloth

This project provides a complete setup for fine-tuning the Qwen 1.5B model using Unsloth for efficient training. The setup includes data preparation, model fine-tuning, and inference capabilities.

## Features

- **LoRA Fine-tuning**: Uses Unsloth with LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning
- **Multiple LoRA Configurations**: Pre-configured settings for different hardware and requirements
- **4-bit Quantization**: Memory-efficient training with Unsloth optimizations
- **CSV Dataset Support**: Loads data from CSV files with System/User/Assistant format
- **Easy Configuration**: Simple configuration for training parameters
- **Inference Script**: Test your fine-tuned model with interactive prompts
- **Sample Dataset**: Includes a sample machine learning Q&A dataset
- **Hardware Analysis**: Automatic hardware detection and configuration recommendations

## Project Structure

```
fine-tune-qwen/
├── requirements.txt              # Python dependencies
├── sample_dataset.csv            # Sample dataset for testing
├── fine_tune_qwen.py             # Basic fine-tuning script
├── fine_tune_qwen_lora.py        # Enhanced LoRA fine-tuning script
├── lora_config.py                # LoRA configuration presets
├── choose_lora_config.py         # Interactive LoRA config chooser
├── inference.py                  # Inference script for testing
├── create_dataset.py             # Dataset creation utility
├── setup.py                      # Automated setup script
├── install_dependencies.py       # Specialized dependency installer
├── quick_start.py                # Guided quick start process
└── README.md                     # This file
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies** (choose one method):

   **Method 1: Automated installation (Recommended)**
   ```bash
   python install_dependencies.py
   ```

   **Method 2: Manual installation**
   ```bash
   # First install PyTorch 2.7+ (required for Unsloth)
   pip install torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Then install other dependencies
   pip install -r requirements.txt
   ```

   **Method 3: Quick setup**
   ```bash
   python setup.py
   ```

3. **Optional: Install CUDA for GPU acceleration**
   - Make sure you have CUDA installed if you want to use GPU acceleration
   - The script will automatically detect and use GPU if available

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

### 2. Choose LoRA Configuration (Recommended)

```bash
python choose_lora_config.py
```

This will analyze your hardware and recommend the best LoRA configuration.

### 3. Fine-tune the Model

**Option A: Enhanced LoRA Fine-tuning (Recommended)**
```bash
python fine_tune_qwen_lora.py
```

**Option B: Basic Fine-tuning**
```bash
python fine_tune_qwen.py
```

**Configuration Options** (modify in `fine_tune_qwen_lora.py`):
- `MODEL_NAME`: Model to fine-tune (default: "Qwen/Qwen1.5-1.8B")
- `DATASET_PATH`: Path to your CSV dataset
- `LORA_CONFIG_NAME`: LoRA configuration (default: "balanced")
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `BATCH_SIZE`: Training batch size (default: 2)

### 3. Test the Fine-tuned Model

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

## Troubleshooting

### Common Issues

1. **PyTorch Version Error**
   - **Error**: `requirement torch>=2.7`
   - **Solution**: Use the automated installation script:
     ```bash
     python install_dependencies.py
     ```
   - **Alternative**: Install PyTorch 2.7+ manually first:
     ```bash
     pip install torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

2. **Out of Memory Errors**
   - Reduce `BATCH_SIZE` in the script
   - Reduce `MAX_SEQ_LENGTH`
   - Use gradient accumulation (already configured)

3. **Slow Training**
   - Ensure you're using a GPU
   - Check if CUDA is properly installed
   - Consider reducing the dataset size for testing

4. **Model Loading Errors**
   - Ensure all dependencies are installed
   - Check internet connection for model download
   - Verify the model path is correct

5. **Unsloth Installation Issues**
   - Make sure PyTorch 2.7+ is installed first
   - Try installing from source: `pip install git+https://github.com/unslothai/unsloth.git`
   - Check if you have enough disk space (Unsloth is large)

### Performance Tips

- Use a GPU for significantly faster training
- Start with a small dataset to test the setup
- Monitor training progress with wandb (optional)
- Adjust learning rate based on your dataset size

## Customization

### Using Different Models

To use a different Qwen model, change the `MODEL_NAME` in `fine_tune_qwen.py`:

```python
MODEL_NAME = "Qwen/Qwen1.5-7B"  # For 7B model
MODEL_NAME = "Qwen/Qwen1.5-14B" # For 14B model
```

### Adjusting Training Parameters

Modify the training arguments in the `create_training_arguments` function:

```python
training_args = create_training_arguments(
    output_dir="./my-model",
    num_train_epochs=5,           # More epochs
    per_device_train_batch_size=4, # Larger batch size
    learning_rate=1e-4,           # Different learning rate
)
```

## Sample Output

After fine-tuning, you can test the model:

```
Question: What is machine learning?
Response: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.

Question: How do you handle missing data?
Response: Missing data can be handled through several methods: 1) Deletion (removing rows or columns with missing values), 2) Imputation (filling missing values with mean, median, mode, or predicted values), 3) Using algorithms that handle missing data natively, or 4) Creating a separate category for missing values.
```

## License

This project is for educational and research purposes. Please ensure you comply with the license terms of the Qwen model and Unsloth library.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this fine-tuning setup.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning optimizations
- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [Hugging Face](https://huggingface.co/) for the transformers library 