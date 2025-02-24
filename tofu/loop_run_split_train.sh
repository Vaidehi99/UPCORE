#!/bin/bash

# Arrays of values for split_eval and split_model
split_model_values=()
for i in {3..5}; do
    split_model_values+=("cf_cluster_${i}_subsampled")
done

# split_eval_values=("${split_model_values[@]}")

for i in {3..5}; do
    split_eval_values+=("cf_cluster_${i}")
done
 # Add your split_model values here

# Check if both arrays have the same length
if [ "${#split_eval_values[@]}" -ne "${#split_model_values[@]}" ]; then
    echo "Error: split_eval_values and split_model_values arrays must have the same length."
    exit 1
fi

# Config file to source
config_file="bash_config.conf"  # Base config file that contains other parameters


Loop over the indices of the arrays
for i in "${!split_eval_values[@]}"; do
    # Set the current split_eval and split_model
    split_eval="${split_eval_values[$i]}"
    split_model="${split_model_values[$i]}"

    echo "Running with split_eval=${split_eval} and split_model=${split_model}"

    # Source the base config file
    if [ ! -f "$config_file" ]; then
        echo "Config file $config_file not found!"
        exit 1
    fi
    source "$config_file"

    CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node=1 --master_port=12337 forget.py --config-name=forget_${forget_loss}.yaml split=$split_model batch_size=4 gradient_accumulation_steps=4 model_family=$model_family lr=$lr




done

echo "All evaluations completed!"
