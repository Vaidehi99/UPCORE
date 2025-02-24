#!/bin/bash

# Check if the config file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

# Source the config file
config_file=$1
if [ ! -f "$config_file" ]; then
    echo "Config file not found!"
    exit 1
fi

source "$config_file"

# Run the forget.py script
# CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.run --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget_${forget_loss}.yaml split=$split_model batch_size=4 gradient_accumulation_steps=4 model_family=$model_family lr=$lr

input_path="../cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/${forget_loss}_${lr}_${split_model}_${num_epoch}/" #"../cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/${forget_loss}_${lr}_${split}_${num_epoch}/"


# Find all checkpoint files, extract the checkpoint numbers, and get the highest one
# Find the highest checkpoint from the files in the input path

for file in "$input_path"checkpoint-*; do
    # Check if the file exists
    if [ -e "$file" ]; then
        # Extract the checkpoint number
        if [[ $file =~ checkpoint-([0-9]+) ]]; then
            checkpoint_number="${BASH_REMATCH[1]}"
            # Update highest_checkpoint if the current number is higher
            if (( checkpoint_number > highest_checkpoint )); then
                highest_checkpoint=$checkpoint_number
            fi
        fi
    else
        echo "No checkpoint files found in $input_path"
        exit 1
    fi
done

if [ -n "$highest_checkpoint" ]; then
    model_path="$input_path/checkpoint-$highest_checkpoint"
    echo "Highest checkpoint model path: $model_path"
else
    echo $input_path
    echo $highest_checkpoint
    echo "No checkpoints found."
fi

# Run the evaluation script
CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_util.py model_family=$model_family split=$split_eval model_path=$model_path

# Aggregate evaluation statistics
python aggregate_eval_stat.py retain_result=$model_path/eval_results/ds_sizeFalse/eval_log_aggregated.json ckpt_result=$model_path/eval_results/ds_sizeFalse/eval_log_aggregated.json method_name=base save_file=${split_eval}_${forget_loss}_result.csv
