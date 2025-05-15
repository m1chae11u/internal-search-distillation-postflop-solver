"""
Usage:
    python do_sft/peft_sft.py \\
        --model_name MODEL_NAME \\
        --traindata_path DATASET_PATH \\
        --output_dir OUTPUT_PATH \\
        [--batch_size BATCH_SIZE] \\
        [--learning_rate LEARNING_RATE] \\
        [--num_epochs NUM_EPOCHS] \\
        [--max_length MAX_LENGTH] \\
        [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] \\
        [--lora_r LORA_R] \\
        [--lora_alpha LORA_ALPHA] \\
        [--lora_dropout LORA_DROPOUT] \\
        [--gpu_id GPU_ID] # Specify the single GPU ID to use

This script implements a simple single-GPU PEFT SFT pipeline using LoRA to finetune LLMs,
allowing selection of the target GPU.

Example Usage using local model (Llama-3.1-8B-Instruct):

python -m do_sft.peft_sft \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --traindata_path /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/train_turn_river_sets/50k_turn_river_search_tree_datasubset_train.json \
    --output_dir /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/do_sft/sft_model_weights/turn_river_50k_peft_sft \
    --gpu_id 1
"""

import os
import sys
import argparse
import json
import logging
import torch
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from tqdm import tqdm
from dotenv import load_dotenv

from .helper_functions import (
    setup_hf_env, find_local_model_path, load_dataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a language model using PEFT (LoRA) on a specific single GPU.')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Base model name to fine-tune')
    parser.add_argument('--traindata_path', type=str, required=True,
                       help='Path to JSON training dataset file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Number of steps for gradient accumulation')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='The specific single GPU ID to use for training (e.g., 0, 1, 2).')
    return parser.parse_args()

def setup_gpu_environment(target_gpu_id: int):
    """Configure environment to use a specific single GPU."
    """
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    logger.info(f"Total available GPUs: {device_count}")

    if not 0 <= target_gpu_id < device_count:
        logger.error(f"Invalid gpu_id {target_gpu_id}. Available GPUs are 0 to {device_count - 1}.")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)
    logger.info(f"Set CUDA_VISIBLE_DEVICES='{target_gpu_id}'. PyTorch will see this as cuda:0.")

    if torch.cuda.device_count() != 1:
        logger.warning(f"Expected 1 visible GPU after setting CUDA_VISIBLE_DEVICES, but found {torch.cuda.device_count()}. Check environment.")
    try:
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} (Original ID: {target_gpu_id}, Visible as cuda:{torch.cuda.current_device()})")
    except Exception as e:
        logger.error(f"Failed to set device to cuda:0 (targeted GPU {target_gpu_id}): {e}")
        sys.exit(1)

def prepare_dataset(dataset_path, tokenizer, max_length):
    """Prepare and tokenize dataset for training.
    
    Args:
        dataset_path (str): Path to the training dataset
        tokenizer: Tokenizer to process the texts
        max_length (int): Maximum sequence length
        
    Returns:
        Dataset: HuggingFace Dataset ready for training
    """
    raw_data = load_dataset(dataset_path)
    
    processed_data = []
    for item in raw_data:
        if "input" in item and "output" in item:
            prompt = item["input"]
            completion = item["output"]
            
            formatted_text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ], tokenize=False)
            
            processed_data.append({"text": formatted_text})
    
    dataset = Dataset.from_list(processed_data)
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None, 
        )
        
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    logger.info(f"Dataset processed and tokenized: {len(tokenized_dataset)} examples")
    return tokenized_dataset

def load_base_model(model_name):
    """Load base model and tokenizer for single-GPU PEFT, strictly from local path."""
    local_model_path = find_local_model_path(model_name)
    
    if local_model_path is None:
        logger.error(f"Model '{model_name}' not found locally by find_local_model_path. "
                     f"Please ensure it is downloaded to the expected cache directory. Exiting.")
        sys.exit(1)
    
    logger.info(f"Attempting to load model strictly from local path: {local_model_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path, 
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token")
        
        logger.info(f"Loading base model from local path: {local_model_path} onto 'cuda:0' (the selected single GPU).")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path, 
            device_map="cuda:0", 
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )
    except EnvironmentError as e:
        logger.error(f"Failed to load model from local path '{local_model_path}'. "
                     f"Ensure the model is correctly downloaded and accessible. Error: {e}")
        sys.exit(1)
    
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def train_model(args, model, tokenizer, train_dataset):
    """Fine-tune model using LoRA.
    
    Args:
        args: Command line arguments
        model: Base model to fine-tune
        tokenizer: Tokenizer for the model
        train_dataset: Processed training dataset
        
    Returns:
        Fine-tuned model
    """
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=False,
        fp16=True,
        deepspeed=None,
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return model

def main():
    args = parse_arguments()
    set_seed(args.seed)
    setup_gpu_environment(args.gpu_id)
    setup_hf_env()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading base model: {args.model_name}")
    model, tokenizer = load_base_model(args.model_name)
    
    logger.info(f"Processing training data from: {args.traindata_path}")
    train_dataset = prepare_dataset(args.traindata_path, tokenizer, args.max_length)
    
    trained_model = train_model(args, model, tokenizer, train_dataset)
    logger.info(f"Fine-tuning completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
