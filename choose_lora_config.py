#!/usr/bin/env python3
"""
Interactive LoRA Configuration Chooser
This script helps you choose the right LoRA configuration for your setup.
"""

import torch
from lora_config import print_available_configs, get_lora_config, explain_lora_parameters

def check_hardware():
    """Check available hardware and provide recommendations."""
    print("🔍 Hardware Analysis")
    print("=" * 30)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_memory = []
        
        for i in range(gpu_count):
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_memory.append(memory)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({memory:.1f} GB)")
        
        max_memory = max(gpu_memory)
        print(f"\nTotal GPU Memory: {max_memory:.1f} GB")
        
        if max_memory >= 24:
            return "high_end"
        elif max_memory >= 12:
            return "mid_range"
        elif max_memory >= 8:
            return "low_end"
        else:
            return "very_limited"
    else:
        print("No GPU detected - will use CPU")
        return "cpu_only"

def get_recommendations(hardware_level):
    """Get LoRA configuration recommendations based on hardware."""
    recommendations = {
        "high_end": {
            "primary": "performance",
            "description": "High-end GPU (24GB+) - Use performance configuration",
            "configs": ["performance", "balanced", "efficient"]
        },
        "mid_range": {
            "primary": "balanced", 
            "description": "Mid-range GPU (12-24GB) - Use balanced configuration",
            "configs": ["balanced", "efficient", "performance"]
        },
        "low_end": {
            "primary": "efficient",
            "description": "Low-end GPU (8-12GB) - Use efficient configuration",
            "configs": ["efficient", "balanced", "minimal"]
        },
        "very_limited": {
            "primary": "minimal",
            "description": "Very limited GPU (<8GB) - Use minimal configuration",
            "configs": ["minimal", "efficient"]
        },
        "cpu_only": {
            "primary": "minimal",
            "description": "CPU-only training - Use minimal configuration",
            "configs": ["minimal"]
        }
    }
    
    return recommendations[hardware_level]

def interactive_choice():
    """Interactive configuration choice."""
    print("\n🎯 LoRA Configuration Chooser")
    print("=" * 40)
    
    # Check hardware
    hardware_level = check_hardware()
    recommendations = get_recommendations(hardware_level)
    
    print(f"\n💡 Recommendation: {recommendations['description']}")
    print(f"   Primary choice: {recommendations['primary']}")
    
    # Show available configurations
    print("\n📋 Available Configurations:")
    print_available_configs()
    
    # Get user choice
    while True:
        print(f"\n🤔 Choose your LoRA configuration:")
        print(f"   Recommended: {recommendations['primary']}")
        print(f"   Other options: {', '.join(recommendations['configs'])}")
        
        choice = input(f"\nEnter configuration name (or 'help' for explanations): ").strip().lower()
        
        if choice == 'help':
            print("\n📚 LoRA Parameter Explanations:")
            explain_lora_parameters()
            continue
        
        if choice in recommendations['configs']:
            config = get_lora_config(choice)
            print(f"\n✅ Selected: {config.name}")
            print(f"   Rank: {config.r}")
            print(f"   Alpha: {config.lora_alpha}")
            print(f"   Learning Rate: {config.learning_rate}")
            return choice
        else:
            print(f"❌ Invalid choice. Please choose from: {', '.join(recommendations['configs'])}")

def generate_config_code(choice):
    """Generate code snippet for the chosen configuration."""
    config = get_lora_config(choice)
    
    code_snippet = f"""
# LoRA Configuration for {config.name}
LORA_CONFIG_NAME = "{choice}"

# Or modify the configuration directly in fine_tune_qwen_lora.py:
# Change this line:
# LORA_CONFIG_NAME = "balanced"
# To:
# LORA_CONFIG_NAME = "{choice}"
"""
    
    return code_snippet

def main():
    """Main function."""
    print("🚀 Qwen 1.5B LoRA Configuration Assistant")
    print("=" * 50)
    
    # Get user choice
    choice = interactive_choice()
    
    # Generate configuration code
    code = generate_config_code(choice)
    
    print("\n" + "=" * 50)
    print("🎉 Configuration Selected!")
    print("=" * 50)
    
    print(code)
    
    print("📝 Next Steps:")
    print("1. Copy the configuration name above")
    print("2. Edit fine_tune_qwen_lora.py and change LORA_CONFIG_NAME")
    print("3. Run: python fine_tune_qwen_lora.py")
    
    print(f"\n💡 Your choice '{choice}' is optimized for your hardware!")
    
    # Additional tips
    config = get_lora_config(choice)
    if config.r <= 8:
        print("\n💾 Memory-saving tips:")
        print("   - Consider reducing batch size if you run out of memory")
        print("   - Use gradient accumulation for larger effective batch sizes")
    elif config.r >= 32:
        print("\n⚡ Performance tips:")
        print("   - You can increase batch size for faster training")
        print("   - Consider using RSLoRA for better stability")

if __name__ == "__main__":
    main() 