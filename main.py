from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from datetime import datetime
from monitor import MultiGPUMonitor
from evolution import EvolutionaryPromptGenerator, EvolutionaryPromptGeneratorConfig
import json
import os

# Constants for model configuration
MODEL_PATH = "/home/alex/Desktop/projectCompliance/Llama-3.1-8B-Instruct-hf"
TENSOR_PARALLEL_SIZE = 4  # Use only 2 GPUs instead of all 4
GPU_MEMORY_UTILIZATION = 0.9  # Reduce from default 0.9 to 0.7
MAX_MODEL_LEN = 120000  #

MAX_TOKENS=10000
TEMPERATURE=0.7
TOP_P=0.9
GENERATIONS = 4

MAX_EVO_GEN = 100
STARTING_POP = 1000
STARTING_LEN = 100
MUTATE = True
POP_MUTATION_RATE = 0.1
PROMPT_MUTATION_RATE = 0.1

def main():
    #Step 1: Initialize model and tokenizer 
        # Load the model

    timestamp = int(time.time())
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_date_{date_str}_{timestamp}"
    run_folder = f"evolution_results/{run_id}"
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    SAVE_PATH = f"./{run_folder}"

    llm = LLM(model = MODEL_PATH,
                dtype="float16",
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,  # Use only 2 GPUs instead of all 4
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,  # Reduce from default 0.9 to 0.7
                max_model_len=MAX_MODEL_LEN,  # Reduce from 131072 to fit in available KV cache
                disable_log_stats=False  # Enable stats logging
            )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast = True)
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    # Initialize monitoring
    monitor = MultiGPUMonitor(interval=0.025)

    evolution_config = EvolutionaryPromptGeneratorConfig(starting_pop = STARTING_POP,
                                                        max_generation=MAX_EVO_GEN,
                                                        starting_len = STARTING_LEN,
                                                        fitness_func = "latency_tokpsec",  # "latency", "throughput", "latency_power"
                                                        evolution = "random_breeding",
                                                        num_sets = 10,  # number of sets to select from
                                                        mutate = MUTATE,
                                                        pop_mutation_rate = POP_MUTATION_RATE,
                                                        prompt_mutation_rate = PROMPT_MUTATION_RATE
                                                        )

    evolution = EvolutionaryPromptGenerator(model = llm,
                                            tokenizer = tokenizer,
                                            sampling_params = sampling_params,
                                            monitor = monitor,
                                            config = evolution_config,
                                            save_path = f"{SAVE_PATH}",
                                            )

    print(f"Starting Evolution with {MAX_EVO_GEN} Generations")
    evolution.run()

    # data to be output
    # json of all generations with prompts, token ids, and response fitness function and fitness
    # json of all generation and prompts with time series of gpu usage, latency, throughput, etc.

    print(f"Evolution results saved to {SAVE_PATH}")
    print(f"Completed Run with {MAX_EVO_GEN} Generations")
    return

if __name__ == "__main__":
    main()