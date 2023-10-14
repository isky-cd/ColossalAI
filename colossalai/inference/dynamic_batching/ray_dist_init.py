import logging
import os

import ray
import ray.util.collective as collective
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.shardformer import ShardConfig
from colossalai.testing import free_port

from colossalai.inference.manager import start_dynamic_batching
from colossalai.inference.dynamic_batching.ray_init_config  import EngineArgsClass, RooterArgsClass

ray_serve_logger = logging.getLogger("ray.serve")

def log_cuda_info(scope_name: str):
    ray_serve_logger.info(f" {scope_name}: ray.get_gpu_ids(): {ray.get_gpu_ids()}")
    ray_serve_logger.info(
        f" {scope_name}: CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'NO DEVICES FOUND!')}"
    )
    if torch.cuda.is_available():
        ray_serve_logger.info(
            f" {scope_name}: cuda current_device: {torch.cuda.current_device()}, cuda device count: {torch.cuda.device_count()}"
        )
    else:
        ray_serve_logger.info(f" {scope_name}: cuda is not available!")

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, model_path: str, tensor_parallel_size: int, max_batch_size: int, max_input_len: int, max_output_len: int, router_config: RooterArgsClass):
        log_cuda_info("Worker.init")
        self.tensor_parallel_size = tensor_parallel_size
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.router_config = router_config

    def setup(self, world_size, rank, port):
        
        # initialize a ray collective group, otherwise colossalai distributed env won't be built successfully
        collective.init_collective_group(world_size, rank, "nccl", "default")
        # initialize and set distributed environment
        colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
        ray_serve_logger.info(f"Worker with rank {rank} (world size {world_size}) setting up..")
        log_cuda_info("Worker.setup")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, pad_token_id=self.tokenizer.pad_token_id, torch_dtype=torch.float16
        )

        shard_config = ShardConfig(enable_tensor_parallelism=True if world_size > 1 else False, inference_only=True)
        self.infer_engine = TPInferEngine(
            self.model, shard_config, self.max_batch_size, self.max_input_len, self.max_output_len
        )
        self.start_dynamic_batching = start_dynamic_batching(self.router_config, self.infer_engine, [])

        return True

    def generate(self, request_id, prompt, sampling_params) -> str:
        
        ray_serve_logger.info(f"text: {prompt}")

        results_generator = self.start_dynamic_batching.generate(prompt, sampling_params, request_id)

        final_output = None
        for request_output in results_generator:
            final_output = request_output

        assert final_output is not None
        ray_serve_logger.info(f"Generated text: {final_output}")
        return final_output

class Driver:
    def __init__(self, router_config: RooterArgsClass, engine_config: EngineArgsClass):
        log_cuda_info("Driver:init")
        model_path = engine_config.model
        tensor_parallel_size = engine_config.tensor_parallel_size

        self.num_workers = tensor_parallel_size
        self.workers = []
        init_rets = []

        # Just grab a free port on localhost
        # NOTE workers in this communication group listen to the same port
        available_port = free_port()

        for i in range(self.num_workers):
            worker_name = "worker_idx_{}".format(i)
            w = Worker.options(name=worker_name).remote(
                model_path, self.num_workers, engine_config.max_batch_size, engine_config.max_input_len, engine_config.max_output_len, router_config
            )
            self.workers.append(w)
            init_rets.append(w.setup.remote(self.num_workers, i, available_port))
        _options = {
            "group_name": "default_driver",
            "world_size": self.num_workers,
            "ranks": [i for i in range(self.num_workers)],
            "backend": "nccl",
        }
        collective.create_collective_group(self.workers, **_options)
        _ = ray.get(init_rets)

    # set batch wait delay in seconds and maximum number of sequences in a batch
    def generate(self, request_id, prompt, sampling_params):
        results = ray.get([w.generate.remote(request_id, prompt, sampling_params) for w in self.workers])
        text_res = results[0]  # get any one of the copies
        return text_res