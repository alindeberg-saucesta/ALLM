#!/bin/bash
set -e

# Script to run all pre-training end-to-end.
# This script will first download training and validation data
# and then start running the pre-training loop.

export PYTHONPATH="${PYTHONPATH}:$PWD/src"

if [[ "$(basename "$PWD")" != "alLLM" ]]; then
    echo "Error: You are not in the 'alLLM' project directory. Current directory is '$PWD'."
    exit 1
fi

# Create log file for script, training logic handles its own logging
LOGFILE="$PWD/temp_data/logs/run_training_e2e_$(date +'%Y_%m_%d-%H_%M_%Z').log"
mkdir -p "$(dirname "$LOGFILE")" && [ -f "$LOGFILE" ] || touch "$LOGFILE"
: > "$LOGFILE" # Empty the log file
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp $*" | tee -a "$LOGFILE"
}

log "LOGFILE: $LOGFILE"
log "DEBUG_MODE is set to $DEBUG_MODE"

if [ "$DEBUG_MODE" == "True" ]; then
    export LOG_LEVEL="DEBUG"
else
    export LOG_LEVEL="INFO"
fi
log "Python log level set to $LOG_LEVEL"

# Check if running on GPU-instance
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    log "Running on NVIDIA GPU-instance!"
    pip install transformers && pip install tiktoken && pip install --upgrade scipy && pip install --upgrade networkx && pip install datasets && pip install tqdm && pip install einops && pip install "numpy<2"
fi

# Download pretraining data
if [ ! -d "$PWD/temp_data/edu_fineweb10B" ]; then
    log "Downloading Pre-Training dataset."
    python src/data_processing/data_downloader.py
    log "Done downloading Pre-Training dataset."
else
    log "Skipping downloading Pre-Training dataset."
fi

# Start pretraining
log "Start training MyLLM."
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log "Number of GPUs: $GPU_COUNT"
    torchrun --standalone --nproc_per_node=$GPU_COUNT pretrain.py
else
    python pretrain.py
fi
log "Done training aLLM."
