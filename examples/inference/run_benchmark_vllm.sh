ROOT=$(realpath $(dirname $0))
PY_SCRIPT=${ROOT}/benchmark_vllm.py
GPU=$(nvidia-smi -L | head -1 | cut -d' ' -f4 | cut -d'-' -f1)

mkdir -p logs

# benchmark llama2-7b one single GPU

CUDA_VISIBLE_DEVICES_set_n_least_memory_usage() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 1

export CUDA_VISIBLE_DEVICES=6

for in_len in 512 1024; do
    for bsz in 16 32 64; do
        python3 ${PY_SCRIPT} --model /llama --batch-size $bsz --input-len $in_len --output-len 256 | tee logs/${GPU}_in${in_len}_out256_bs${bsz}_vllm_profile.txt
    done
done
