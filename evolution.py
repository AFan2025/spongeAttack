import time
from sympy import fu
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from monitor import MultiGPUMonitor
import random
import torch
import numpy as np
import json

class EvolutionaryPromptGeneratorConfig:
    def __init__(self, starting_pop = 100, 
                 max_generation=100, starting_len = 10, 
                 fitness_func = "latency", 
                 evolution = "basic", 
                 num_sets = 10, 
                 mutate = False, 
                 pop_mutation_rate = 0.1,
                 prompt_mutation_rate = 0.1):
        self.starting_pop = starting_pop
        self.max_generation = max_generation
        self.starting_len = starting_len
        self.num_sets = num_sets
        self.mutate = mutate
        self.pop_mutation_rate = pop_mutation_rate
        self.prompt_mutation_rate = prompt_mutation_rate

        self.fitness_func = fitness_func # e.g., "latency", "throughput"

        self.evolution = evolution # e.g., "basic", "random breeding", "crossover", "append", "mutation"

class EvolutionaryPromptGenerator:
    def __init__(self, model, tokenizer, sampling_params, monitor, config: EvolutionaryPromptGeneratorConfig, save_path = None):
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.monitor = monitor

        self.config = config
        self.starting_pop = config.starting_pop
        self.max_generation = config.max_generation
        self.vocab_size = tokenizer.vocab_size

        self.current_generation = 0
        self.current_prompts = self.initialize_prompts()

        if save_path:
            self.save_path = save_path
        else:
            raise ValueError("please specify a save path to an existing folder")
        self.fitness_save_path = f"{self.save_path}/evolution_fitness.jsonl"
        self.gpu_data_save_path = f"{self.save_path}/evolution_gpu_data.jsonl"

        self.config_save_path = f"{self.save_path}/config.jsonl"

        with open(self.config_save_path, 'w') as f_config:
            f_config.write(json.dumps(vars(self.config)) + "\n")
            f_config.flush()

    def initialize_prompts(self):
        prompts = []
        for _ in range(self.starting_pop):
            prompt = self.generate_random_prompt()
            prompts.append(prompt)
        return prompts

    def generate_random_prompt(self):
        # Generate random token IDs in the vocabulary range
        random_token_ids = torch.randint(0, self.vocab_size, (self.config.starting_len,)).tolist()
        prompt = self.tokenizer.decode(random_token_ids, skip_special_tokens=True)

        return (prompt, random_token_ids)

    def run(self):
        generations = self.max_generation
        for gen in range(generations):
            self.current_generation = gen
            print(f"RUNNING Generation {gen+1}/{generations}")
            generation_results = []
            idx = 0

            # Evaluate current prompts
            for prompt, token_ids in self.current_prompts:

                # output is a dict of latency, tokens_per_second, peak GPU usage, etc.
                result = self.test_load(prompt)

                fitness = self.fitness_function(result)
                # print(f"Prompt result: {result}")

                saved_results = {"prompt": prompt,
                                           "token_ids": token_ids,
                                           "result": result,
                                           "fitness": fitness}

                # Save best prompts to history
                generation_results.append(saved_results)
                
                self.stream_generation_data(saved_results, idx)
                idx += 1

            # Log generation results
            # print(f"Current Generation {gen} Population: {len(generation_results)}")
            # self.history[f"generation{gen}"] = generation_results

            # Select best prompts
            best_prompts = self.tournament_selection(generation_results,
                                                    num_sets=self.config.num_sets,
                                                    filter=0.5)
            print(f"Current Generation {gen+1}: Best Prompts: {len(best_prompts)}")

            # self.stream_generation_data(generation_results)

            # Evolve prompts for next generation
            new_generation_prompts = []
            self.current_prompts = best_prompts

            for prompt_info in best_prompts:
                token_ids = prompt_info[1]
                # print(f"token ids: {token_ids}")
                # Apply mutation or crossover to create new prompts
                new_prompts = self.evolve_prompt(token_ids)
                # print(new_prompts)
                # print(new_generation_prompts)
                new_generation_prompts.extend(new_prompts)

            print(f"New Generation {gen+2} Size: {len(new_generation_prompts)}")

            self.current_prompts = new_generation_prompts
        # return self.history
        return

    def test_load(self, prompt):
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful AI assistant. Please answer the following question as best you can."},
             {"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )

        self.monitor.start()
        start = time.time()
        outputs = self.model.generate([formatted_prompt], self.sampling_params)
        end = time.time()
        self.monitor.stop()

        total_time = end - start
        output = outputs[0]
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        tokens_per_second = num_tokens / total_time if total_time > 0 else 0

        gpu_stats = self.monitor.get_stats()
        average_power = sum([gpu_stats[f"GPU_{i}"]["max_power_W"] for i in range(self.monitor.gpu_count)]) / self.monitor.gpu_count
        self.monitor.clear_stats()

        return {
            "total_time": total_time,
            "num_tokens": num_tokens,
            "tokens_per_second": tokens_per_second,
            "generated_text": generated_text,
            "gpu_stats": gpu_stats,
            "avg_gpu_power": average_power,
            "avg_gpu_util": sum([gpu_stats[f"GPU_{i}"]["max_gpu_util"] for i in range(self.monitor.gpu_count)]) / self.monitor.gpu_count,
            "avg_mem_used_MB": sum([gpu_stats[f"GPU_{i}"]["max_mem_used_MB"] for i in range(self.monitor.gpu_count)]) / self.monitor.gpu_count
        }
    
    def stream_generation_data(self, result, idx):
        with open(self.fitness_save_path, 'a') as f_fitness:
            unique_id = f"{self.current_generation}.{idx}"
            f_fitness.write(json.dumps({
                "generation": self.current_generation,
                    "unique_id": unique_id,
                    "fitness": result["fitness"],
                    "latency": result["result"]["total_time"],
                    "num_tokens": result["result"]["num_tokens"],
                    "tokens_per_second": result["result"]["tokens_per_second"],
                    "avg_gpu_power": result["result"]["avg_gpu_power"],
                    "avg_gpu_util": result["result"]["avg_gpu_util"],
                    "avg_mem_used_MB": result["result"]["avg_mem_used_MB"],
                    "prompt": result["prompt"],
                    "token_ids": result["token_ids"],
                    "generated_text": result["result"]["generated_text"],
                }) + "\n")
            f_fitness.flush()
        with open(self.gpu_data_save_path, 'a') as f_gpu:
            # for idx, result in enumerate(generation_results):
            unique_id = f"{self.current_generation}.{idx}"
            f_gpu.write(json.dumps({
                "generation": self.current_generation,
                "unique_id": unique_id,
                "timestamp": result["result"]["gpu_stats"].get("GPU_1")["timestamp"] if result["result"].get("GPU_1") else None,
                "GPU1": result["result"]["gpu_stats"].get("GPU_1", {}),
                "GPU2": result["result"]["gpu_stats"].get("GPU_2", {}),
                "GPU3": result["result"]["gpu_stats"].get("GPU_3", {}),
                "GPU4": result["result"]["gpu_stats"].get("GPU_4", {}),
            }) + "\n")
            f_gpu.flush()
        return

    def tournament_selection(self, prompts, num_sets=10, filter=0.5):
        selected = []
        for _ in range(num_sets):
            tournament = random.sample(prompts, k=max(2, len(self.current_prompts)//num_sets))
            tournament = sorted(tournament, key=lambda x: x["fitness"], reverse=True)
            # if random.random() < filter:
            for i in range(int(len(tournament)*filter)):
                selected.append((tournament[i]["prompt"], tournament[i]["token_ids"]))
            # else:
                # selected.append(random.choice(tournament[1:]))
        return selected

    def fitness_function(self, result):
        """
        Fed results in the form of a dictionary with keys:
        total_time, num_tokens, tokens_per_second, generated_text, gpu_stats
        define custom fitness function here
        """
        if self.config.fitness_func == "latency_power":
            return result["total_time"] * result["avg_gpu_power"]  # Higher is better
        elif self.config.fitness_func == "latency":
            return result["total_time"]
        elif self.config.fitness_func == "throughput":
            return result["tokens_per_second"]
        elif self.config.fitness_func == "latency_tokpsec":
            return result["total_time"] * 100 + result["num_tokens"]
        else:
            raise ValueError(f"Unknown fitness function: {self.config.fitness_func}")

    def evolve_prompt(self, token_ids):
        evolved_prompts = []
        new_prompts = []
        if self.config.evolution == "basic":
            # Simple mutation: randomly change some tokens
            for _ in range(2):
                new_token_ids = token_ids.copy()
                for _ in range(max(1, len(new_token_ids) // 10)):
                    idx = random.randint(0, len(new_token_ids) - 1)
                    new_token_ids[idx] = random.randint(0, self.vocab_size - 1)
                # new_prompt = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                evolved_prompts.append(new_token_ids)
        elif self.config.evolution == "crossover":
            # Randomly breed two prompts
            for _ in range(2):
                parent1 = random.choice(self.current_prompts)[1]
                parent2 = random.choice(self.current_prompts)[1]
                crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                new_token_ids = parent1[:crossover_point] + parent2[crossover_point:]
                # new_prompt = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                evolved_prompts.append(new_token_ids)
        elif self.config.evolution == "random_breeding":
            # Randomly breed two prompts using binary mask (numpy version)
            for _ in range(2):
                parent1 = random.choice(self.current_prompts)[1]
                # Ensure both parents have the same length or pad to match
                if len(parent1) != len(token_ids):
                    raise ValueError("Parents must have same length")
                # max_len = max(len(parent1), len(parent2))
                # parent1_padded = np.pad(parent1, (0, max_len - len(parent1)), constant_values=0)
                # parent2_padded = np.pad(parent2, (0, max_len - len(parent2)), constant_values=0)
                
                # Create random binary mask (0 = parent1, 1 = parent2)
                binary_mask = np.random.randint(0, 2, size=len(parent1))
                
                # Apply mask using vectorized operations
                new_token_ids = np.where(binary_mask == 0, token_ids, parent1).tolist()
                
                # new_prompt = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                evolved_prompts.append(new_token_ids)
        # for _ in range(2):
        #     new_token_ids = token_ids.copy()
        #     # Randomly mutate some tokens
        #     for _ in range(max(1, len(new_token_ids) // 10)):
        #         idx = random.randint(0, len(new_token_ids) - 1)
        #         new_token_ids[idx] = random.randint(0, self.vocab_size - 1)
        #     new_prompt = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        #     new_prompts.append((new_prompt, new_token_ids))
        for prompt in evolved_prompts:
            new_token_ids = prompt
            # Apply random mutation
            if self.config.mutate:
                if random.random() < self.config.pop_mutation_rate:
                    new_token_ids = self.mutate_tokens(new_token_ids)
            new_prompt = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            new_prompts.append((new_prompt, new_token_ids))
        return new_prompts
    
    def mutate_tokens(self, token_ids):
        new_token_ids = token_ids.copy()
        for _ in range(max(1, int(len(new_token_ids) * self.config.prompt_mutation_rate))):
            idx = random.randint(0, len(new_token_ids) - 1)
            new_token_ids[idx] = random.randint(0, self.vocab_size - 1)
        return new_token_ids

    def get_generation(self):
        return self.current_generation

    def add_prompt(self, prompt):
        self.history.append(prompt)
