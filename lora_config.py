#!/usr/bin/env python3
"""
LoRA Configuration for Qwen Fine-tuning with Unsloth
This file contains different LoRA configurations and explanations.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoRAConfig:
    """LoRA configuration class."""
    name: str
    description: str
    r: int  # Rank
    lora_alpha: int  # Alpha parameter
    lora_dropout: float  # Dropout rate
    target_modules: List[str]  # Target modules for LoRA
    bias: str  # Bias handling
    use_rslora: bool  # Use RSLoRA
    learning_rate: float  # Learning rate for this config

# Predefined LoRA configurations
LORA_CONFIGS = {
    "efficient": LoRAConfig(
        name="Efficient",
        description="Memory-efficient configuration for limited resources",
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_rslora=False,
        learning_rate=1e-4
    ),
    
    "balanced": LoRAConfig(
        name="Balanced",
        description="Good balance between performance and efficiency (default)",
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_rslora=False,
        learning_rate=2e-4
    ),
    
    "performance": LoRAConfig(
        name="Performance",
        description="Higher performance with more parameters",
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_rslora=True,
        learning_rate=3e-4
    ),
    
    "minimal": LoRAConfig(
        name="Minimal",
        description="Minimal configuration for very limited resources",
        r=4,
        lora_alpha=8,
        lora_dropout=0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        use_rslora=False,
        learning_rate=5e-5
    ),
    
    "custom": LoRAConfig(
        name="Custom",
        description="Custom configuration - modify as needed",
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_rslora=False,
        learning_rate=2e-4
    )
}

def get_lora_config(config_name: str = "balanced") -> LoRAConfig:
    """
    Get a LoRA configuration by name.
    
    Args:
        config_name (str): Name of the configuration
    
    Returns:
        LoRAConfig: The requested configuration
    """
    if config_name not in LORA_CONFIGS:
        print(f"Configuration '{config_name}' not found. Using 'balanced' instead.")
        config_name = "balanced"
    
    return LORA_CONFIGS[config_name]

def print_available_configs():
    """Print all available LoRA configurations."""
    print("Available LoRA Configurations:")
    print("=" * 50)
    
    for name, config in LORA_CONFIGS.items():
        print(f"\n{config.name} ({name}):")
        print(f"  Description: {config.description}")
        print(f"  Rank (r): {config.r}")
        print(f"  Alpha: {config.lora_alpha}")
        print(f"  Dropout: {config.lora_dropout}")
        print(f"  Target Modules: {config.target_modules}")
        print(f"  RSLoRA: {config.use_rslora}")
        print(f"  Learning Rate: {config.learning_rate}")

def apply_lora_config(model, config: LoRAConfig):
    """
    Apply LoRA configuration to a model.
    
    Args:
        model: The model to apply LoRA to
        config (LoRAConfig): LoRA configuration
    
    Returns:
        model: Model with LoRA applied
    """
    from unsloth import FastLanguageModel
    
    print(f"Applying LoRA configuration: {config.name}")
    print(f"  Rank: {config.r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Target modules: {config.target_modules}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )
    
    return model

def explain_lora_parameters():
    """Explain LoRA parameters and their effects."""
    print("LoRA Parameters Explanation:")
    print("=" * 40)
    
    explanations = {
        "r (Rank)": {
            "description": "The rank of the low-rank matrices in LoRA",
            "effects": {
                "Higher": "More parameters, better performance, more memory",
                "Lower": "Fewer parameters, less memory, potentially lower performance"
            },
            "recommended_range": "4-64 (8-32 for most cases)"
        },
        
        "lora_alpha": {
            "description": "Scaling factor for LoRA weights",
            "effects": {
                "Higher": "Stronger LoRA influence, faster learning",
                "Lower": "Weaker LoRA influence, more stable training"
            },
            "recommended_range": "8-64 (usually set to r or 2*r)"
        },
        
        "lora_dropout": {
            "description": "Dropout rate for LoRA layers",
            "effects": {
                "Higher": "Better regularization, less overfitting",
                "Lower": "Less regularization, potentially better performance"
            },
            "recommended_range": "0.0-0.2 (0.0-0.1 for most cases)"
        },
        
        "target_modules": {
            "description": "Which layers to apply LoRA to",
            "options": {
                "q_proj, k_proj, v_proj, o_proj": "Attention layers only",
                "gate_proj, up_proj, down_proj": "MLP layers only",
                "All": "Both attention and MLP layers"
            },
            "recommendation": "Start with attention layers, add MLP if needed"
        },
        
        "use_rslora": {
            "description": "Use RSLoRA (Rank Stabilized LoRA)",
            "effects": {
                "True": "Better stability, potentially better performance",
                "False": "Standard LoRA, faster training"
            },
            "recommendation": "Use for larger models or when stability is important"
        }
    }
    
    for param, info in explanations.items():
        print(f"\n{param}:")
        print(f"  {info['description']}")
        print("  Effects:")
        for effect, description in info['effects'].items():
            print(f"    {effect}: {description}")
        if 'recommended_range' in info:
            print(f"  Recommended range: {info['recommended_range']}")
        if 'recommendation' in info:
            print(f"  Recommendation: {info['recommendation']}")

if __name__ == "__main__":
    print_available_configs()
    print("\n" + "=" * 50)
    explain_lora_parameters() 