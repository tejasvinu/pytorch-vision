#!/bin/bash

# --- Job Configuration (Adapt for your HPC Scheduler, e.g., Slurm) ---
#SBATCH --job-name=cifar_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # Master process for torchrun
#SBATCH --cpus-per-task=16   # Number of CPUs for data loaders (e.g., 4 per GPU if using 2 GPUs)
#SBATCH --gres=gpu:2        # Request 2 GPUs. Adjust as needed (e.g., gpu:1 for single GPU)
#SBATCH --mem=32G           # CPU memory
#SBATCH --time=20:00:00     # Max 2 hours for CIFAR. Adjust as needed.
#SBATCH --output=cifar_train_job_%j.out
#SBATCH --error=cifar_train_job_%j.err
#SBATCH --partition=gpu     # Or your specific GPU partition

# --- Environment Setup ---
echo "Loading modules..."
module purge
module load anaconda3/anaconda3
module load cuda/11.8

source /home/apps/anaconda3/etc/profile.d/conda.sh  # Use the correct path you found

conda activate pytorch-gpu

module load oneapi/vtune/latest # For Vtune profiling (optional)

echo "Modules loaded:"
module list
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" # Should be set by Slurm if gres is used

# --- Directory and Experiment Setup ---
VISION_DIR="/home/poweropt1/tej/pytorch-vision" # IMPORTANT: Set this path!
SCRIPT_PATH="${VISION_DIR}/references/classification/train.py"
OUTPUT_DIR_BASE="./cifar_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR_BASE}"

# --- nvidia-smi Logging ---
GPU_LOG_FILE="${OUTPUT_DIR_BASE}/gpu_stats.csv"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Starting nvidia-smi logging to ${GPU_LOG_FILE}"
    nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw --format=csv -l 5 -f "${GPU_LOG_FILE}" &
    NVIDIA_SMI_PID=$!
else
    echo "No GPUs detected by Slurm (CUDA_VISIBLE_DEVICES not set), skipping nvidia-smi."
    NVIDIA_SMI_PID=""
fi


# --- Training Command (Multi-GPU with torchrun if NUM_GPUS > 1) ---
# Determine number of GPUs available (Slurm sets CUDA_VISIBLE_DEVICES)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
else
    NUM_GPUS=0 # Default to 0 if no GPUs are assigned
fi
echo "Number of GPUs requested/detected: $NUM_GPUS"


# Model and Training Parameters
CHOSEN_DATASET="CIFAR10" # Change to "CIFAR100" if desired
MODEL="resnet18"          # Good starting point for CIFAR (resnet34, resnet50 also work)
                          # VGG models or smaller custom CNNs can also be tested.
EPOCHS=50                 # CIFAR trains faster; 50 epochs can show good convergence
LEARNING_RATE=0.1         # Common starting LR for ResNet on CIFAR
BATCH_SIZE_PER_GPU=256    # CIFAR images are small; can often use larger batch sizes
                          # Adjust based on GPU VRAM and model size

# Output directory for this specific run
RUN_OUTPUT_DIR="${OUTPUT_DIR_BASE}/model_output_${CHOSEN_DATASET}_${MODEL}"
mkdir -p "${RUN_OUTPUT_DIR}"

echo "Starting PyTorch ${CHOSEN_DATASET} training..."
echo "Model: ${MODEL}, Epochs: ${EPOCHS}, Batch Size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Output directory: ${RUN_OUTPUT_DIR}"

# Base Python command arguments
BASE_ARGS=" \
    ${SCRIPT_PATH} \
    --model ${MODEL} \
    --dataset ${CHOSEN_DATASET} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE_PER_GPU} \
    --workers $((${SLURM_CPUS_PER_TASK:-4} / ($NUM_GPUS > 0 ? $NUM_GPUS : 1) )) \
    --lr ${LEARNING_RATE} \
    --output-dir ${RUN_OUTPUT_DIR} \
    --amp" # Enable Automatic Mixed Precision
    # --resume ${RUN_OUTPUT_DIR}/checkpoint.pth # To resume

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using ${NUM_GPUS} GPUs with torchrun."
    # Use a known IP address for the launch node if available, or localhost if single node
    # Slurm typically provides SLURM_LAUNCH_NODE_IPADDR or similar
    # If not, and it's a single node job, 127.0.0.1 is fine for rdzv_endpoint.
    RDZV_ENDPOINT="${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}:29500"

    PYTHON_COMMAND="torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${RDZV_ENDPOINT} ${BASE_ARGS}"
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Using 1 GPU (no torchrun needed for single GPU, direct script execution)."
    PYTHON_COMMAND="/home/apps/anaconda3/envs/pytorch-gpu/bin/python ${BASE_ARGS} --device cuda:0"
else # NUM_GPUS is 0 or not set
    echo "No GPUs specified, running on CPU (if supported by script, or will likely fail if script requires GPU)."
    PYTHON_COMMAND="/home/apps/anaconda3/envs/pytorch-gpu/bin/python ${BASE_ARGS} --device cpu"
fi


# Choose one profiling option if needed:
# Option 1: Run Python script directly
CMD_TO_RUN="${PYTHON_COMMAND}"

# Option 2: Run with Intel Vtune for HPC Performance Analysis
# VTUNE_RESULT_DIR="${OUTPUT_DIR_BASE}/vtune_hpc_${CHOSEN_DATASET}_${MODEL}"
# mkdir -p "${VTUNE_RESULT_DIR}"
# CMD_TO_RUN="vtune -collect hpc-performance -result-dir \"${VTUNE_RESULT_DIR}\" -quiet -- ${PYTHON_COMMAND}"

echo "Executing: ${CMD_TO_RUN}"
eval "${CMD_TO_RUN}" # Use eval if your command has complex quoting or variables

TRAIN_EXIT_CODE=$?
echo "Training finished with exit code: ${TRAIN_EXIT_CODE}"

# --- Cleanup ---
if [ -n "$NVIDIA_SMI_PID" ]; then
    echo "Stopping nvidia-smi logging..."
    kill ${NVIDIA_SMI_PID}
    wait ${NVIDIA_SMI_PID} 2>/dev/null
fi

echo "Job finished. Output and logs are in ${OUTPUT_DIR_BASE}"
# if [ -d "${VTUNE_RESULT_DIR}" ]; then
#     echo "Vtune results (if collected) are in ${VTUNE_RESULT_DIR}"
# fi

exit ${TRAIN_EXIT_CODE}