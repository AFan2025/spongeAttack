from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

# Load the model
llm = LLM(model="/home/alex/Desktop/projectCompliance/Llama-3.1-8B-Instruct-hf",
            dtype="float16",
            tensor_parallel_size=2,  # Use only 2 GPUs instead of all 4
            gpu_memory_utilization=0.7,  # Reduce from default 0.9 to 0.7
            max_model_len=120000,  # Reduce from 131072 to fit in available KV cache
            disable_log_stats=False  # Enable stats logging
        )

# Load tokenizer to format chat messages
tokenizer = AutoTokenizer.from_pretrained("/home/alex/Desktop/projectCompliance/Llama-3.1-8B-Instruct-hf")

# Create chat messages
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

# Format the chat using the tokenizer's chat template
formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print("Formatted prompt:")
print(formatted_prompt)
print("-" * 50)

sampling_params = SamplingParams(max_tokens=100, temperature=0.7)

start = time.time()
outputs = llm.generate([formatted_prompt], sampling_params)
end = time.time()

# Calculate timing metrics
total_time = end - start
output = outputs[0]
generated_text = output.outputs[0].text
num_tokens = len(output.outputs[0].token_ids)

print(f"Total inference time: {total_time:.3f}s")
# print(f"Tokens generated: {num_tokens}")
# print(f"Tokens per second: {num_tokens / total_time:.2f}")
# print(f"Time per token: {total_time / num_tokens * 1000:.2f}ms")
# print("-" * 50)
print("outputs objects", vars(output.outputs[0]))
print("Generated text:")
print(generated_text)