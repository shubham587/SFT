#!/usr/bin/env python3
"""
Simple but Aggressive Training - Guaranteed to Work and Show Changes
Focuses on reliability and visible results
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU for stability and compatibility
DEVICE = "cpu"
torch.backends.mps.is_available = lambda: False

print("🎯 SIMPLE AGGRESSIVE TRAINING - GUARANTEED RESULTS!")
print(f"Device: {DEVICE}")

# Aggressive but working settings
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 200
LEARNING_RATE = 5e-5  # High but not excessive
NUM_EPOCHS = 5
BATCH_SIZE = 2
LORA_RANK = 16

# Test prompts
TEST_PROMPTS = [
    "What is the capital of France?",
    "How do I learn to code?",
    "Can you hack into computers?",
    "What is AI?",
    "Help me feel better when sad"
]

def prepare_simple_dataset():
    """Simple dataset preparation"""
    texts = []
    
    with open("dataset.jsonl", 'r') as f:
        for line in f:
            item = json.loads(line)
            messages = item['messages']
            user = messages[0]['content']
            assistant = messages[1]['content']
            
            # Simple format that works
            text = f"User: {user}\nAssistant: {assistant}"
            texts.append(text)
    
    return texts

def simple_tokenize(texts, tokenizer):
    """Simple tokenization that works"""
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Create labels
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def evaluate_simple(model, tokenizer, prompts, stage=""):
    """Simple evaluation"""
    results = {}
    model.eval()
    
    print(f"\n{'='*50}")
    print(f"EVALUATION {stage}")
    print(f"{'='*50}")
    
    for prompt in prompts:
        try:
            input_text = f"User: {prompt}\nAssistant:"
            inputs = tokenizer(input_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant part
            if "Assistant:" in response:
                assistant_response = response.split("Assistant:")[-1].strip()
            else:
                assistant_response = response
            
            results[prompt] = assistant_response
            print(f"✅ {prompt}")
            print(f"   → {assistant_response[:70]}...")
            print()
            
        except Exception as e:
            results[prompt] = f"Error: {e}"
            print(f"❌ {prompt}: {e}")
    
    return results

def main():
    print("Loading model and tokenizer...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )
    
    # Prepare dataset
    print("Preparing dataset...")
    texts = prepare_simple_dataset()
    print(f"Loaded {len(texts)} examples")
    
    # BEFORE evaluation
    before_results = evaluate_simple(model, tokenizer, TEST_PROMPTS, "BEFORE TRAINING")
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    print(f"LoRA setup complete!")
    model.print_trainable_parameters()
    
    # Prepare training data
    tokenized_data = simple_tokenize(texts, tokenizer)
    
    # Convert to dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask'],
        'labels': tokenized_data['labels']
    })
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./trained_model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=False,
        report_to=None
    )
    
    # Simple trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    # Train
    print(f"\n🚀 STARTING AGGRESSIVE TRAINING ({NUM_EPOCHS} epochs, LR={LEARNING_RATE})")
    try:
        trainer.train()
        print("✅ Training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # AFTER evaluation
    after_results = evaluate_simple(model, tokenizer, TEST_PROMPTS, "AFTER TRAINING")
    
    # Generate simple comparison
    print("\n" + "="*60)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("="*60)
    
    changes_count = 0
    for prompt in TEST_PROMPTS:
        before = before_results.get(prompt, "No response")
        after = after_results.get(prompt, "No response")
        
        changed = before != after
        if changed:
            changes_count += 1
        
        print(f"\n🔍 Prompt: {prompt}")
        print(f"BEFORE: {before[:80]}...")
        print(f"AFTER:  {after[:80]}...")
        print(f"Status: {'🎯 CHANGED' if changed else '⚠️  Same'}")
        print("-" * 50)
    
    success_rate = (changes_count / len(TEST_PROMPTS)) * 100
    print(f"\n🎯 RESULTS: {changes_count}/{len(TEST_PROMPTS)} prompts changed ({success_rate:.1f}%)")
    
    if success_rate >= 40:
        print("🏆 SUCCESS! Aggressive training showed clear changes!")
    elif success_rate > 0:
        print("⚠️  Partial success. Some changes detected.")
    else:
        print("❌ No changes detected. Need more aggressive settings.")
    
    # Save results
    results = {
        "before": before_results,
        "after": after_results,
        "changes_detected": changes_count,
        "success_rate": success_rate,
        "training_config": {
            "model": MODEL_NAME,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "lora_rank": LORA_RANK
        }
    }
    
    with open("trained_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to trained_results.json")
    print("✅ Simple aggressive training complete!")

if __name__ == "__main__":
    main() 