"""Benchmark the latency of processing a single batch of requests."""
"""Modified from https://github.com/ModelTC/lightllm/blob/main/test/model/model_infer.py
"""
import argparse
import time
import numpy as np
import torch
import torch.distributed as dist
import colossalai.utils.device as device_utils
from contextlib import nullcontext
# from colossalai.testing import free_port

from lightllm.models.llama.model import LlamaTpPartModel

GIGABYTE = 1024**3


def run_lightllm_inference(model: LlamaTpPartModel, input_data: torch.Tensor, batch_size: int, input_len: int, output_len: int, verbose: bool = False):
    if verbose:
        print("Can use mem size:", model.mem_manager.can_use_mem_size)
        print("Can use req size:", model.req_manager.can_use_req_size)
    
    start_time = time.perf_counter()
    prefill_start_time = time.perf_counter()
    
    # Prepare indexes
    b_req_idx = model.req_manager.alloc(batch_size).to(torch.int32)
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len
    total_token_num = batch_size * input_len
    # Prefill stage
    logits = model.forward(
        batch_size, 
        total_token_num, 
        input_len, 
        input_data,
        b_req_idx, 
        b_start_loc, 
        b_seq_len, 
        is_prefill=True
    )
    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach()

    torch.cuda.synchronize()
    if verbose:
        print("prefill time cost:", (time.perf_counter() - prefill_start_time) * 1000)

    # Generation stage
    for i in range(output_len):
        torch.cuda.synchronize()
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1

        logits = model.forward(
            batch_size, 
            total_token_num, 
            input_len + i + 1, 
            predict_ids.cuda().reshape(-1),
            b_req_idx, 
            b_start_loc, 
            b_seq_len, 
            is_prefill=False
        )
        prob_out = torch.softmax(logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach()
        
    torch.cuda.synchronize()

    model.mem_manager.free_all()
    model.req_manager.free_all()

    end_time = time.perf_counter()

    latency = end_time - start_time
    if verbose:
        print(f"Latency: {latency} seconds")
        print(f"Throughput: {batch_size * output_len / latency} tokens/s")

    return latency


def benchmark(args: argparse.Namespace):

    dist.init_process_group("nccl", init_method=f"tcp://{args.host}:{args.port}", rank=args.rank, world_size=args.world_size)
    
    batch_size = args.batch_size
    input_len = args.input_len
    output_len = args.output_len

    model_kvargs = {
        "tp_rank": args.rank,
        "world_size": args.world_size,
        "weight_dir": args.model_path,
        "max_total_token_num": batch_size * (input_len + output_len),
        "load_way": "HF",
        "mode": [],  # triton_int8kv
        "max_req_num": batch_size,
        "max_seq_length": (input_len + output_len)
    }

    lightllm_model = LlamaTpPartModel(model_kvargs)

    dummy_prompt_token_ids = [[0] * input_len] * batch_size
    dummy_prompt_token_ids_flattened = [elem for seq in dummy_prompt_token_ids for elem in seq]
    dummy_data = torch.tensor(dummy_prompt_token_ids_flattened).cuda()
    
    N_WARMUP_STEPS = 2
    
    ctx = (
        torch.profiler.profile(
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=N_WARMUP_STEPS, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./lightllm_tb_log_" + str(batch_size)+ "_" + str(input_len)),
        )
        if args.profile
        else nullcontext()
    )
    
    latencies = []
    
    with ctx:
        for _ in range(N_WARMUP_STEPS):
            run_lightllm_inference(lightllm_model, dummy_data, batch_size, input_len, output_len)
            if args.profile:
                ctx.step()
        latencies.append(run_lightllm_inference(lightllm_model, dummy_data, batch_size, input_len, output_len, verbose=True))
        if args.profile:
            ctx.step()        
    
    avg_latency = np.mean(latencies)
    avg_token_latency = avg_latency / (batch_size * output_len)
    msg = ""
    msg += "\n-------Perf Summary-------\n"
    msg += f"Whole batch latency (end2end): {avg_latency * 1000:.2f} ms\n"
    msg += f"Whole batch per token latency: {avg_token_latency * 1000:.2f} ms\n"
    msg += f"Throughput: {batch_size * output_len / avg_latency} tokens/s\n"
    if torch.cuda.is_available():
        msg += f"-------Memory Summary Device: {device_utils.current_device()}-------\n"
        msg += f"Max memory allocated: {device_utils.max_memory_allocated() / GIGABYTE:.2f} GB\n"
        msg += f"Max memory reserved: {device_utils.max_memory_reserved() / GIGABYTE:.2f} GB\n"
    
    print(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=24999, help="Port number")
    parser.add_argument("--rank", type=int, default=0, help="Rank id")
    parser.add_argument("--world_size", type=int, default=1, help="World size")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Model path to llama-7b")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--input_len", type=int, default=32, help="Input length")
    parser.add_argument("--output_len", type=int, default=128, help="Output length")
    parser.add_argument("--num_iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--profile", default=False, action="store_true", help="enable torch profiler")

    args = parser.parse_args()

    benchmark(args)