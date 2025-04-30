# Default values
method="nerfacto"
gpu_id=0

# Parse arguments (GPU and method)
method="${1:-$method}"
gpu_id="${2:-$gpu_id}"

echo "Processing all plots scenes"
echo "Method: $method; using GPU: $gpu_id"

export CUDA_VISIBLE_DEVICES="$gpu_id"

# Ensure CUDA devices are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA device not detected!"
    exit 1
fi

# Set paths
base_folder="/workspace/Wheat-GS"
dataset_folder="$base_folder/dataset/Wheat-GS-data-nerf/20240717"
output_path="$base_folder/outputs"
transforms_json=transforms_with_splits.json
results_json=test_results.json

# Iterate through all input folders
for input_folder in "$dataset_folder"/*; do
    if [ ! -d "$input_folder" ]; then
        continue
    fi
    scene=$(basename $input_folder)
    scene_path="$output_path/$scene"
    model_path="$scene_path/$method"
    
    if [ -d "$model_path" ]; then
        latest_exp_name=$(ls $model_path | sort -r | head -n 1)
        exp_path="$model_path/$latest_exp_name"
        if [ -f "$exp_path/$results_json" ]; then
            echo "Metrics for scene $scene are already processed on $exp_path"
            continue
        fi
    fi
        
    echo "Processing scene $scene"
    
    # Train
    echo "Training scene $scene"
    ns-train $method \
        --data "$input_folder/$transforms_json" \
        --pipeline.model.camera-optimizer.mode off \
        --vis wandb \
        --project-name WheatGS-nerfstudio \
        nerfstudio-data \
        --train-split-fraction 1.0

    # Get directory
    latest_exp_name=$(ls $model_path | sort -r | head -n 1)
    exp_path="$model_path/$latest_exp_name"

    # Render and calculate metrics
    # echo "Rendering run: $exp_path"
    ns-render dataset --load_config "$exp_path/config.yml" --split train+test --output-path "$exp_path"
    ns-eval --load-config "$exp_path/config.yml" --output-path "$exp_path/$results_json"
    #python scripts/metrics.py $exp_path/test

done
echo "Done processing all scenes"

