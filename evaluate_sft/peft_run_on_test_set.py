'''
peft_run_on_test_set.py
run this before eval_pipeline.py

This script runs a PEFT (LoRA) fine-tuned SFT model on a test set 
using Hugging Face transformers and the PEFT library.

Example usage on 20k train data:

    python -m evaluate_sft.peft_run_on_test_set \
        --base_model_path /srv/share/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct \
        --adapter_path /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/do_sft/sft_model_weights/turn_river_50k_peft_sft \
        --test_set_path /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/test_turn_river_sets/4788_turn_river_search_tree_datasubset_test.json \
        --output_path /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/evaluate_sft/50k_peft_3.1-8b/test_predictions.json \
        --gpu_id 1 \
        --batch_size 100 \
        --max_new_tokens 512
'''
import os
import sys
import argparse
import json
import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add project root to sys.path to allow imports from do_sft
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from do_sft.helper_functions import find_local_model_path, load_dataset, setup_hf_env
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from do_sft.helper_functions import find_local_model_path, load_dataset, setup_hf_env

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a PEFT SFT model on a test set using Transformers and PEFT.')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Name or path of the base model (e.g., "meta-llama/Llama-3.1-8B-Instruct").')
    parser.add_argument('--adapter_path', type=str, required=True,
                        help='Path to the trained LoRA adapter directory.')
    parser.add_argument('--test_set_path', type=str, required=True,
                        help='Path to the JSON test dataset file.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output JSON file with predictions.')
    
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='The specific GPU ID to use (e.g., 0, 1). Set to -1 for CPU.')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load the base model in 4-bit precision using BitsAndBytes.')
    parser.add_argument('--merge_adapter', action='store_true',
                        help='Merge the adapter into the base model before inference (requires more memory).')

    # Generation Parameters
    parser.add_argument('--batch_size', type=int, default=4, # Smaller default due to potentially higher memory use
                        help='Batch size for inference.')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature. 0.0 for greedy decoding.')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p (nucleus) sampling parameter.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter for generation.')

    return parser.parse_args()

def setup_device(target_gpu_id: int):
    """Sets up the device for PyTorch operations."""
    if target_gpu_id == -1:
        logger.info("Using CPU for inference.")
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        if 0 <= target_gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{target_gpu_id}")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(target_gpu_id)} (cuda:{target_gpu_id})")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id) # Also set for safety, though .to(device) is primary
            return device
        else:
            logger.warning(f"GPU ID {target_gpu_id} is invalid. Available GPUs: {torch.cuda.device_count()}. Falling back to cuda:0.")
            return torch.device("cuda:0")
    else:
        logger.warning("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

def load_peft_model_and_tokenizer(
    base_model_name_or_path: str,
    adapter_path: str,
    device: torch.device,
    load_in_4bit: bool = False,
    merge_adapter: bool = False
):
    """Loads the base model, applies PEFT adapter, and tokenizer."""
    logger.info(f"Attempting to find local path for base model: {base_model_name_or_path}")
    local_base_model_path = find_local_model_path(base_model_name_or_path)

    if local_base_model_path is None:
        logger.warning(
            f"Base model '{base_model_name_or_path}' not found locally by find_local_model_path. "
            f"Attempting to load '{base_model_name_or_path}' directly. This may involve downloading if it's a HuggingFace Hub ID."
        )
        model_load_target = base_model_name_or_path
    else:
        logger.info(f"Using local base model path: {local_base_model_path}")
        model_load_target = local_base_model_path

    logger.info(f"Loading base model from: {model_load_target}")
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        logger.info("4-bit quantization enabled.")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_load_target,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if not load_in_4bit else None, # float16 if not quantizing, let BNB handle if quantizing
        trust_remote_code=True,
        # device_map handled by .to(device) later, unless merging immediately without specific device mapping strategy
    )

    tokenizer = AutoTokenizer.from_pretrained(model_load_target, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Important for decoder-only models
        logger.info("Set pad_token to eos_token and padding_side to left for tokenizer.")

    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    # If not merging, load adapter onto the same device as the model will be.
    # If merging, PeftModel can handle device placement during merge if base_model isn't on target device yet.
    model = PeftModel.from_pretrained(base_model, adapter_path) #, device_map can be added here if needed before merge

    if merge_adapter:
        logger.info("Merging adapter into base model...")
        try:
            model = model.merge_and_unload()
            logger.info("Adapter merged successfully.")
        except Exception as e:
            logger.warning(f"Could not merge adapter: {e}. Proceeding with unmerged PEFT model.")
    
    logger.info(f"Moving model to device: {device}")
    model.to(device)
    model.eval()
    logger.info("PEFT model and tokenizer loaded successfully.")
    return model, tokenizer

def main():
    args = parse_arguments()
    setup_hf_env()
    device = setup_device(args.gpu_id)

    model, tokenizer = load_peft_model_and_tokenizer(
        base_model_name_or_path=args.base_model_path,
        adapter_path=args.adapter_path,
        device=device,
        load_in_4bit=args.load_in_4bit,
        merge_adapter=args.merge_adapter
    )

    logger.info(f"Loading test dataset from: {args.test_set_path}")
    test_data = load_dataset(args.test_set_path)
    if not test_data:
        logger.error("Failed to load test data or test data is empty. Exiting.")
        sys.exit(1)

    results = []
    current_batch_prompts = []
    current_batch_original_data = []

    logger.info(f"Starting inference on {len(test_data)} examples with batch size {args.batch_size}...")

    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True, # args.temperature > 0 usually implies do_sample=True
        "temperature": args.temperature if args.temperature > 0 else None, # None for greedy if temp is 0
        "top_p": args.top_p,
        "top_k": args.top_k,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    if args.temperature == 0:
        generation_params["do_sample"] = False
        # For greedy, often temperature, top_p, top_k are not needed or set to specific values
        # Transformers handles this if do_sample is False.

    logger.info(f"Using generation parameters: {generation_params}")

    for i, item in enumerate(tqdm(test_data, desc="Processing test set")):
        game_state = item.get("input")
        ground_truth = item.get("output")

        if game_state is None:
            logger.warning(f"Skipping item {i} due to missing 'input' field.")
            results.append({"input": None, "output": ground_truth, "predicted_output": "ERROR: Missing input field"})
            continue

        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": game_state}],
            tokenize=False,
            add_generation_prompt=True
        )
        current_batch_prompts.append(formatted_prompt)
        current_batch_original_data.append(item)

        if len(current_batch_prompts) == args.batch_size or i == len(test_data) - 1:
            if not current_batch_prompts:
                continue
            
            inputs = tokenizer(current_batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings - args.max_new_tokens - 5).to(device) # Ensure space for generation
            
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_params)
                
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for idx, full_decoded_text in enumerate(decoded_outputs):
                    # Extract only the generated part (after the prompt)
                    # This assumes the prompt is part of the decoded output.
                    # This logic might need adjustment based on how tokenizer.batch_decode and model.generate behave together.
                    # A common way is to slice based on input_ids length if they are part of `outputs` or available.
                    # For now, let's find the original prompt in the output and take text after it.
                    original_prompt_in_output = current_batch_prompts[idx]
                    if original_prompt_in_output in full_decoded_text:
                         # This is a simple way, might fail if prompt is subtly changed by tokenization/detokenization
                         # A more robust way is to decode only `outputs[j, inputs.input_ids.shape[1]:]`
                        predicted_text = full_decoded_text.split(original_prompt_in_output, 1)[-1].strip()
                    else:
                        # Fallback if exact prompt not found; decode generated tokens only
                        # This requires knowing the length of input tokens for each item in the batch
                        input_ids_for_item = tokenizer(current_batch_prompts[idx], return_tensors="pt").input_ids
                        num_input_tokens = input_ids_for_item.shape[1]
                        generated_tokens = outputs[idx][num_input_tokens:]
                        predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    original_item = current_batch_original_data[idx]
                    results.append({
                        "input": original_item.get("input"),
                        "output": original_item.get("output"),
                        "predicted_output": predicted_text
                    })

            except Exception as e:
                logger.error(f"Error during generation for a batch: {e}", exc_info=True)
                for original_item in current_batch_original_data:
                    results.append({
                        "input": original_item.get("input"),
                        "output": original_item.get("output"),
                        "predicted_output": f"ERROR: Generation failed - {str(e)}"
                    })
            
            current_batch_prompts = []
            current_batch_original_data = []

    logger.info(f"Inference completed. Total results: {len(results)}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving results to: {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main() 