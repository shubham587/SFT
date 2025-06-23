# SFT (Supervised Fine-Tuning) Project

A simple and aggressive approach to fine-tuning language models using LoRA (Low-Rank Adaptation) with guaranteed visible results.

## 🎯 Project Overview

This project demonstrates supervised fine-tuning of the **SmolLM2-135M** model using a custom dataset. The training approach is designed to be:
- **Simple**: Easy to understand and modify
- **Aggressive**: Optimized for visible changes in model behavior
- **Reliable**: Built with error handling and fallback mechanisms
- **Mac-optimized**: Specifically tested on Apple Silicon (M1/M2/M3)

## 📁 Project Structure

```
SFT/
├── train.py                          # Main training script
├── dataset.jsonl                     # Training dataset (25 examples)
├── requirements.txt                  # Python dependencies
├── trained_model/                    # Output directory for trained models
├── trained_results.json             # Training results and comparisons
├── before_after_mac_smollm-135m-mac.md  # Detailed results analysis
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS with Apple Silicon (M1/M2/M3) or compatible system
- 8GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Training

```bash
python train.py
```

The script will:
1. Load the SmolLM2-135M model
2. Evaluate the model on test prompts (BEFORE training)
3. Apply LoRA fine-tuning with aggressive settings
4. Re-evaluate the model (AFTER training)
5. Generate a comparison report
6. Save results to `trained_results.json`

## 🔧 Configuration

### Model Settings
- **Base Model**: `HuggingFaceTB/SmolLM2-135M` (135M parameters)
- **Device**: CPU (optimized for stability)
- **Precision**: `torch.float32`

### Training Parameters
- **LoRA Rank**: 16
- **Learning Rate**: 5e-5
- **Epochs**: 5
- **Batch Size**: 2
- **Max Length**: 200 tokens

### Test Prompts
The model is evaluated on these 5 test prompts:
1. "What is the capital of France?"
2. "How do I learn to code?"
3. "Can you hack into computers?"
4. "What is AI?"
5. "Help me feel better when sad"

## 📊 Results

### Latest Training Results
- **Success Rate**: 100% (5/5 prompts showed changes)
- **Model**: SmolLM2-135M
- **Training Time**: ~5-10 minutes on Mac
- **Memory Usage**: ~2-4GB RAM

### Before vs After Examples

**Prompt**: "What is the capital of France?"

**Before Training:**
```
The capital of France is Paris.
What is the correct answer?
A. Paris
Which of the following countries is not a member of NATO?
A. France
...
```

**After Training:**
```
Paris is the capital of France.
What is the capital of France?
More on the capital of France
What is the capital of France?
Why is the capital of France the capital of France?
...
```

## 📚 Dataset

The training dataset (`dataset.jsonl`) contains 25 conversation examples covering:
- General knowledge questions
- Educational topics
- Ethical responses (refusing harmful requests)
- Practical advice
- Creative responses

Each example follows the format:
```json
{
  "messages": [
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

## 🛠️ Technical Details

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA alpha parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
    lora_dropout=0.1,        # Dropout rate
    bias="none",             # No bias training
    task_type=TaskType.CAUSAL_LM,  # Causal language modeling
)
```

### Training Arguments
- **Output Directory**: `./trained_model`
- **Logging Steps**: 5
- **Save Strategy**: "no" (to save disk space)
- **FP16**: Disabled for stability
- **Report To**: None (no external logging)

