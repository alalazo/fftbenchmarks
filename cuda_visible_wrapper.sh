#!/bin/bash

NR_GPUS=$(nvidia-smi -L | wc -l)

CURRENT_RANK="${OMPI_COMM_WORLD_RANK:-0}"

CUDA_VISIBLE_DEVICES=$(( ${CURRENT_RANK} % ${NR_GPUS} ))

echo "[INFO]: OpenMPI rank: ${CURRENT_RANK} - using CUDA device: ${CUDA_VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES

$@
