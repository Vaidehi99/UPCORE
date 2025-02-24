#!/bin/bash

# Load topics from JSON using Python
# topics=$(python3 -c "
# import json
# with open('/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json', 'r') as f:
#     topics = json.load(f)
#     topics = topics[:4]+topics[5:6]
# print(json.dumps(topics))  # Ensure valid JSON output
# ")

# # Print the topics to verify
# echo "Selected topics: $topics"

# # Convert JSON array (Python output) into a Bash array
# readarray -t topics_array < <(echo "$topics" | jq -r '.[]')

# # Initialize split_model_values and populate it
# split_model_values=()
# for i in {0..4}; do
#     split_model_values+=("${topics_array[i]}_subsampled")
# done
# for i in {0..4}; do
#     split_eval_values+=("${topics_array[i]}")
# done

split_model_values=()
for i in {5..9}; do
    split_model_values+=("tv_cluster_${i}")
done

# split_eval_values=("${split_model_values[@]}")

for i in {5..9}; do
    split_eval_values+=("tv_cluster_${i}")
done

# Assign split_eval_values from split_model_values
# split_eval_values=("${split_model_values[@]}")

# Debugging: Print arrays to verify
echo "Split model values: ${split_model_values[@]}"
echo "Split eval values: ${split_eval_values[@]}"



# Check if both arrays have the same length
if [ "${#split_eval_values[@]}" -ne "${#split_model_values[@]}" ]; then
    echo "Error: split_eval_values and split_model_values arrays must have the same length."
    exit 1
fi

echo "Split eval values: ${split_eval_values[@]}"


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

    CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.run --nproc_per_node=1 --master_port=12334 forget.py --config-name=forget_${forget_loss}.yaml split=$split_model batch_size=4 gradient_accumulation_steps=4 model_family=$model_family lr=$lr



done

echo "All evaluations completed!"
