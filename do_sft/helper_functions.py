import torch
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
import os
import json
import logging
import time
from tqdm import tqdm
from dotenv import load_dotenv

## Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

## global variables
_tokenizer = None
_openai_model_name = 'gpt-4o'
_client = None
_model = defaultdict(lambda: None)

# Set tensor parallelism environment variable to prevent certain warnings
os.environ["NCCL_DEBUG"] = "WARN"  # Reduce verbosity of NCCL messages

def setup_hf_env():
    """Set up Hugging Face token from config file or .env file."""
    load_dotenv()
    
    if os.getenv('HF_TOKEN'):
        logger.info("Using HF_TOKEN from environment variables")
        return os.getenv('HF_HOME')
    
    try:
        dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
        path_to_config = os.path.join(dir_of_this_script, 'configs', 'config.json')
        
        if os.path.exists(path_to_config):
            with open(path_to_config, 'r') as config_file:
                config_data = json.load(config_file)
            
            if "HF_TOKEN" in config_data:
                os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
                logger.info("Using HF_TOKEN from config file")
                return os.getenv('HF_HOME')
    except Exception as e:
        logger.error(f"Error loading config: {e}")
    
    logger.warning("HF_TOKEN not found in environment or config file")
    return None

def load_model(model_name):
    """Generic function to load a VLLM model."""
    return load_vllm_model(model_name)
    
def get_openai_client():
    """Get an OpenAI client (not used in this version)."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def load_vllm_model(model_name="meta-llama/meta-llama-3.1-8b-instruct"):
    """Load a vLLM model.
    
    Args:
        model_name (str): Model name or path
        
    Returns:
        LLM: A vLLM model instance
    """
    global _model
    if _model[model_name] is None:
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

        # Set environment variable to suppress FlashInfer warning
        os.environ["VLLM_SUPPRESS_FLASHINFER_WARNING"] = "1"

        # Configure options to reduce warnings
        model = LLM(
            model=model_name,  # Use explicit float16 to avoid bfloat16 casting warning
            dtype=torch.float16,
            tensor_parallel_size=1,
            enforce_eager=False,  # Set to False to enable async output processing
            max_model_len=60_000,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,  # Set explicit memory utilization
            # Set swap_space to 0 to avoid warnings about disk space
            swap_space=0,
            # Disable some logging to reduce noise
            disable_log_stats=True,
        )

        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        
        logger.info(f"Model {model_name} loaded. Memory Allocated: {memory_allocated / (1024 ** 3):.2f} GB")
        logger.info(f"Model {model_name} loaded. Memory Reserved: {memory_reserved / (1024 ** 3):.2f} GB")
        _model[model_name] = model
    else:
        model = _model[model_name]
    _ = initialize_tokenizer(model_name) # cache the tokenizer 
    return model

def initialize_tokenizer(model_name=None):
    """Initialize and cache a tokenizer.
    
    Args:
        model_name (str): Model name for the tokenizer
        
    Returns:
        AutoTokenizer: The initialized tokenizer
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer

# New functions for data loading and processing

def load_dataset(dataset_path, limit=None):
    """Load a dataset from a JSON file.
    
    Args:
        dataset_path (str): Path to the JSON dataset file
        limit (int, optional): Limit the number of examples to load
        
    Returns:
        list: List of examples from the dataset
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        if limit and isinstance(limit, int) and limit > 0:
            dataset = dataset[:limit]
            
        logger.info(f"Loaded dataset from {dataset_path} with {len(dataset)} examples" + 
                   (f" (limited to {limit})" if limit else ""))
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset from {dataset_path}: {e}")
        return []

def create_sampling_params(temperature=0.7, max_tokens=100, top_p=0.95):
    """Create and return vLLM sampling parameters.
    
    Args:
        temperature (float): Temperature for sampling
        max_tokens (int): Maximum number of tokens to generate
        top_p (float): Top-p sampling parameter
        
    Returns:
        SamplingParams: vLLM sampling parameters
    """
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

def format_prompt(example, prompt_template=None):
    """Format a prompt using a template and example data.
    
    Args:
        example (dict): Example data from dataset
        prompt_template (str, optional): Template string with placeholders for example fields
        
    Returns:
        str: Formatted prompt
    """
    if prompt_template is None:
        # Default to just returning the 'input' field if it exists
        return example.get('input', str(example))
    
    try:
        # Replace placeholders in template with fields from example
        return prompt_template.format(**example)
    except KeyError as e:
        logger.error(f"Error formatting prompt: missing key {e} in example")
        return str(example)
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}")
        return str(example)

def process_output(raw_output):
    """Process raw model output for evaluation.
    
    Args:
        raw_output: Raw output from the model
        
    Returns:
        str: Processed output text
    """
    if hasattr(raw_output, 'outputs') and len(raw_output.outputs) > 0:
        return raw_output.outputs[0].text.strip()
    return str(raw_output)

def generate_batch(model, prompts, sampling_params, batch_size=8):
    """Generate responses for a batch of prompts.
    
    Args:
        model: The loaded model
        prompts (list): List of prompts to process
        sampling_params: Parameters for sampling
        batch_size (int): Number of prompts to process in one batch
        
    Returns:
        list: List of outputs from the model
    """
    # Process in manageable batches
    all_outputs = []
    
    # Split prompts into batches of size batch_size
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    total_batches = len(prompt_batches)
    logger.info(f"Processing {len(prompts)} prompts in {total_batches} batches of size {batch_size}")
    
    # Process each batch
    for i, batch in enumerate(prompt_batches):
        try:
            logger.debug(f"Processing batch {i+1}/{total_batches} with {len(batch)} prompts")
            start_time = time.time()
            
            # Generate responses for the batch
            batch_outputs = model.generate(batch, sampling_params)
            
            end_time = time.time()
            logger.debug(f"Batch {i+1} completed in {end_time - start_time:.2f}s")
            
            # Add outputs to the list
            all_outputs.extend(batch_outputs)
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {e}")
            
            # Fallback to processing one by one if batch fails
            logger.warning("Falling back to processing prompts one by one")
            for prompt in batch:
                try:
                    single_output = model.generate([prompt], sampling_params)
                    all_outputs.extend(single_output)
                except Exception as inner_e:
                    logger.error(f"Error processing single prompt: {inner_e}")
                    # Add None to maintain alignment with input prompts
                    all_outputs.append(None)
    
    return all_outputs

def process_batch_examples(model, examples, prompt_template, sampling_params, batch_size=8):
    """Process a batch of examples using the model.
    
    Args:
        model: The loaded model
        examples (list): List of examples to process
        prompt_template (str): Template for formatting prompts
        sampling_params: Parameters for sampling
        batch_size (int): Number of examples to process in one batch
        
    Returns:
        list: Results for each example in the batch
    """
    # Format prompts for all examples
    prompts = [format_prompt(example, prompt_template) for example in examples]
    
    # Generate responses in batches
    start_time = time.time()
    all_outputs = generate_batch(model, prompts, sampling_params, batch_size)
    end_time = time.time()
    total_time = end_time - start_time
    
    # Process results
    results = []
    for i, example in enumerate(examples):
        result = example.copy()
        result["prompt"] = prompts[i]
        
        if i < len(all_outputs) and all_outputs[i] is not None:
            output_text = process_output(all_outputs[i])
            result["model_output"] = output_text
        else:
            result["model_output"] = "ERROR: Failed to generate output"
        
        # Add generation time (approximated since we're using batching)
        result["generation_time"] = total_time / len(examples)
        results.append(result)
    
    return results

def find_local_model_path(model_name):
    """Find the path to a local model in the shared HuggingFace directory.
    
    Args:
        model_name (str): Model name in the format "org/model_name"
        
    Returns:
        str or None: Path to the local model if found, None otherwise
    """
    # For models in the format "meta-llama/Llama-3.1-8B-Instruct" 
    if "meta-llama/llama-3.1" in model_name.lower() or "meta-llama/Llama-3.1" in model_name.lower():
        # Handle the specific Llama 3.1 model
        specific_path = "/srv/share/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
        if os.path.exists(specific_path):
            logger.info(f"Found Llama 3.1 model at predefined path: {specific_path}")
            # Find the most recent snapshot
            snapshot_dir = os.path.join(specific_path, 'snapshots')
            if os.path.exists(snapshot_dir):
                snapshots = [d for d in os.listdir(snapshot_dir) if os.path.isdir(os.path.join(snapshot_dir, d))]
                if snapshots:
                    snapshots.sort(reverse=True)
                    latest_snapshot = os.path.join(snapshot_dir, snapshots[0])
                    logger.info(f"Using latest snapshot: {latest_snapshot}")
                    return latest_snapshot
            return specific_path
    
    # Convert model name to the directory format used by HuggingFace
    if "/" in model_name:
        org, model_id = model_name.split("/", 1)
        local_model_dir = f"/srv/share/huggingface/hub/models--{org}--{model_id}"
    else:
        local_model_dir = f"/srv/share/huggingface/hub/models--{model_name}"
    
    try:
        # Check if directory exists and contains model files
        has_model_files = False
        snapshot_dir = os.path.join(local_model_dir, 'snapshots')
        
        if os.path.exists(local_model_dir) and os.path.exists(snapshot_dir):
            # Check if any snapshots contain model files
            for snapshot in os.listdir(snapshot_dir):
                snapshot_path = os.path.join(snapshot_dir, snapshot)
                if os.path.isdir(snapshot_path):
                    if any(f.endswith('.safetensors') or f.endswith('.bin') for f in os.listdir(snapshot_path)):
                        has_model_files = True
                        break
        
        if os.path.exists(local_model_dir) and has_model_files:
            logger.info(f"Found model locally at: {local_model_dir}")
            
            # Find the most recent snapshot
            if os.path.exists(snapshot_dir):
                snapshots = [d for d in os.listdir(snapshot_dir) if os.path.isdir(os.path.join(snapshot_dir, d))]
                if snapshots:
                    # Sort by name (which typically includes timestamps)
                    snapshots.sort(reverse=True)
                    latest_snapshot = os.path.join(snapshot_dir, snapshots[0])
                    logger.info(f"Using latest snapshot: {latest_snapshot}")
                    return latest_snapshot
            
            # If no snapshots found, use the model directory
            return local_model_dir
        else:
            logger.warning(f"Model not found locally at: {local_model_dir}")
            return None
    except Exception as e:
        logger.error(f"Error checking local model: {e}")
        return None