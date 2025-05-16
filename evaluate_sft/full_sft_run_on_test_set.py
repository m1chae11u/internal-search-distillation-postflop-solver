'''
full_sft_run_on_test_set.py
run this before eval_pipeline.py

This script runs a FULL fine-tuned SFT model on a test set using vLLM.

Example Usage:

    python -m evaluate_sft.full_sft_run_on_test_set \
        --model_path /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/do_sft/sft_model_weights/turn_river_20k_full_sft \
        --test_set_path /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/datasets/test_turn_river_sets/4788_turn_river_search_tree_datasubset_test.json \
        --output_path /home/xuandong/mnt/poker/internal-search-distillation-postflop-solver/evaluate_sft/20k_full_3.2-3b/test_predictions.json \
        --gpu_id 2 \
        --batch_size 100 \
        --max_tokens 512

'''
import os
import sys
import argparse
import json
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add project root to sys.path to allow imports from do_sft
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from do_sft.helper_functions import load_dataset, setup_hf_env # find_local_model_path removed as full SFT path is direct
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from do_sft.helper_functions import load_dataset, setup_hf_env

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a full SFT model on a test set using vLLM.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the full SFT model directory.')
    parser.add_argument('--test_set_path', type=str, required=True,
                        help='Path to the JSON test dataset file.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output JSON file with predictions.')
    
    # GPU and vLLM Engine Parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='The specific single GPU ID to target for setup. vLLM will use GPUs set by CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallel size for vLLM.')
    parser.add_argument('--max_model_len', type=int, default=4096,
                        help='Maximum sequence length for the model in vLLM.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.90,
                        help='GPU memory utilization for vLLM (0.0 to 1.0).')

    # Generation Parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference.')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for sampling. 0.0 for greedy decoding.')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p (nucleus) sampling parameter.')
    
    return parser.parse_args()

def setup_gpu_environment_for_eval(target_gpu_id: int):
    """Configure environment to use a specific single GPU for vLLM."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    logger.info(f"Total system GPUs available to PyTorch initially: {device_count}")

    if not 0 <= target_gpu_id < device_count:
        logger.error(f"Invalid gpu_id {target_gpu_id}. Available GPUs are 0 to {device_count - 1}.")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)
    logger.info(f"Set CUDA_VISIBLE_DEVICES='{target_gpu_id}'. vLLM should use this GPU (seen as cuda:0 by vLLM).")
    
    import time
    time.sleep(0.1) 
    visible_gpus_after_set = torch.cuda.device_count()
    if visible_gpus_after_set > 0:
         logger.info(f"PyTorch now sees {visible_gpus_after_set} GPU(s). Current device for PyTorch: {torch.cuda.get_device_name(0)} (relative ID 0).")
    else:
        logger.error(f"No GPUs visible to PyTorch after setting CUDA_VISIBLE_DEVICES. Check CUDA setup and GPU ID {target_gpu_id}.")
        sys.exit(1)

def load_model_and_tokenizer_for_inference(
    model_path: str, # Changed from model_identifier
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90
):
    """Loads the full SFT model and its tokenizer using vLLM."""
    torch.cuda.empty_cache()
    os.environ["VLLM_SUPPRESS_FLASHINFER_WARNING"] = "1"

    logger.info(f"Configuring to load full SFT model from '{model_path}'.")
    logger.info(f"Attempting to load model and tokenizer from '{model_path}' with vLLM.")
    
    llm = LLM(
        model=model_path,
        tokenizer=model_path, 
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        # enable_lora and lora_modules removed as this is for full SFT
        dtype="auto"
    )
    
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        logger.info("Set HuggingFace tokenizer's pad_token to eos_token.")
        
    logger.info("vLLM model and HuggingFace tokenizer loaded successfully.")
    return llm, hf_tokenizer

def main():
    args = parse_arguments()
    setup_hf_env()
    setup_gpu_environment_for_eval(args.gpu_id)

    llm, hf_tokenizer = load_model_and_tokenizer_for_inference(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    logger.info(f"Loading test dataset from: {args.test_set_path}")
    test_data = load_dataset(args.test_set_path)
    if not test_data:
        logger.error("Failed to load test data or test data is empty. Exiting.")
        sys.exit(1)

    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[hf_tokenizer.eos_token_id] if hf_tokenizer.eos_token_id is not None else [],
    )
    logger.info(f"Using sampling parameters: temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    results = []
    prompts_batch = []
    original_data_batch = []

    logger.info(f"Starting inference on {len(test_data)} examples with batch size {args.batch_size}...")

    for i, item in enumerate(tqdm(test_data, desc="Processing test set")):
        game_state = item.get("input")
        ground_truth = item.get("output")

        if game_state is None:
            logger.warning(f"Skipping item {i} due to missing 'input' field: {item}")
            results.append({
                "input": None,
                "output": ground_truth,
                "predicted_output": "ERROR: Missing input field"
            })
            continue

        formatted_prompt = hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": game_state}],
            tokenize=False,
            add_generation_prompt=True 
        )
        
        prompts_batch.append(formatted_prompt)
        original_data_batch.append(item)

        if len(prompts_batch) == args.batch_size or i == len(test_data) - 1:
            if not prompts_batch: # Should not happen if loop has items, but as a safe guard
                continue
            try:
                generated_outputs = llm.generate(prompts_batch, sampling_params)
                
                for idx, gen_output in enumerate(generated_outputs):
                    predicted_text = gen_output.outputs[0].text.strip()
                    original_item = original_data_batch[idx]
                    results.append({
                        "input": original_item.get("input"),
                        "output": original_item.get("output"),
                        "predicted_output": predicted_text
                    })
            except Exception as e:
                logger.error(f"Error during generation for a batch (size {len(prompts_batch)}): {e}")
                for original_item in original_data_batch:
                    results.append({
                        "input": original_item.get("input"),
                        "output": original_item.get("output"),
                        "predicted_output": f"ERROR: Generation failed - {str(e)}"
                    })
            
            prompts_batch = []
            original_data_batch = []

    logger.info(f"Inference completed. Total results: {len(results)}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving results to: {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Attempting to clean up vLLM engine and CUDA cache...")
    try:
        if 'llm' in locals() and llm is not None:
            del llm
            logger.info("vLLM engine deleted.")
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied.")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main() 