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

    # CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node=1 --master_port=12337 forget.py --config-name=forget_${forget_loss}.yaml split=$split_model batch_size=4 gradient_accumulation_steps=4 model_family=$model_family lr=$lr


#     # Update model path dynamically if needed
    model_path="../cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/"

    input_path="../cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/${forget_loss}_${lr}_${split_model}_${num_epoch}/" #"../cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/${forget_loss}_${lr}_${split}_${num_epoch}/"


    checkpoints=()
    checkpoints_found=false

    # Extract checkpoint numbers and store them in an array
    for file in "$input_path"checkpoint-*; do
        if [ -e "$file" ]; then
            if [[ $file =~ checkpoint-([0-9]+) ]]; then
                checkpoint_number="${BASH_REMATCH[1]}"
                checkpoints+=("$checkpoint_number")
                checkpoints_found=true
            fi
        else
            echo "No checkpoint files found in $input_path"
            exit 1
        fi
    done

    # If checkpoints are found, process them
    if [ "$checkpoints_found" = true ]; then
        # Sort the checkpoint numbers in ascending order
        IFS=$'\n' sorted_checkpoints=($(sort -n <<<"${checkpoints[*]}"))
        unset IFS

        echo "Sorted checkpoints: ${sorted_checkpoints[*]}"

        # Select every 5th checkpoint
        for ((i = 0; i < 10; i += 1)); do
            echo "Selected checkpoint: checkpoint-${sorted_checkpoints[i]}"
            model_path="$input_path/checkpoint-${sorted_checkpoints[i]}"
            CUDA_VISIBLE_DEVICES=5 python3 evaluate_util.py model_family=$model_family split=$split_eval model_path=$model_path
            python aggregate_eval_stat.py retain_result=$model_path/eval_results/ds_sizeFalse/eval_log_aggregated.json \
                                  ckpt_result=$model_path/eval_results/ds_sizeFalse/eval_log_aggregated.json \
                                  method_name=base save_file=${split_eval}_${split_model}_${forget_loss}_${sorted_checkpoints[i]}_lora1_c2_${i}_jailbreak_result.csv
            echo "Completed evaluation for split_eval=${split_eval} and split_model=${split_model}_${sorted_checkpoints[i]}"

        done
    else
        echo "No checkpoints matching the criteria were found."
    fi


done

echo "All evaluations completed!"
