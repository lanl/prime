#!/bin/bash
#SBATCH --job-name=pnlp-1_gpu
#SBATCH --output=logs/version_%j/slurm_out/%j.out	             # Redirect standard out to slurm_outs
#SBATCH --error=logs/version_%j/slurm_out/%j.err	             # Redirect standard err to slurm_outs
#SBATCH --partition gpu                                          # GPU partition
#SBATCH --reservation=gpu_debug                                  # Debug
#SBATCH --nodes=1                                                # Number of nodes
#SBATCH --ntasks-per-node=1                                      # Number of processes per node (match GPU count)
#SBATCH --gpus-per-node=1                                        # Number of GPUs per node
#SBATCH --cpus-per-task=4                                        # Adjust CPU count per process
#SBATCH --exclude=nid001432                                      # Exclude node
#SBATCH --exclusive                                              # Use entire node exclusively

# Load environment
source ../../../../../venvs/spike/bin/activate

# Set up environment variables
export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))  # Total processes across nodes
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_LAUNCH_MODE=PARALLEL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=4

# Print debugging info
echo "Launching torchrun with:"
echo "  NODES ($SLURM_JOB_NUM_NODES): $SLURM_JOB_NODELIST"
echo "  SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"

# Run single gpu training/validation/testing
python PL-bert_mlm-esm_init.py

