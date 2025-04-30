# Default values
method="splatfacto"
gpu_id=0
skip_scenes=""

# Function to print usage
usage() {
    echo "Usage: $0 [GPU_ID] [SKIP_SCENES]"
    echo "  GPU_ID: ID of the GPU to use (default: 0)"
    echo "  SKIP_SCENES: Comma-separated list of scenes to skip (default: none)"
    echo "Example: $0 0 \"plot_465,plot_466,plot_467\""
    exit 1
}

# Parse arguments (GPU and scenes to skip)
gpu_id="${1:-$gpu_id}"
skip_scenes="${2:-$skip_scenes}"

# Check for help argument
if [[ "$1" == "--help" ]]; then
    usage
fi

echo "Processing all plots scenes"
echo "Method: $method; using GPU: $gpu_id"
echo "Skipping scenes: $skip_scenes"

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

# Convert skip_scenes to an array
IFS=',' read -r -a skip_scenes_array <<< "$skip_scenes"

# Iterate through all input folders
for input_folder in $(ls -d "$dataset_folder"/* | sort -r); do
    if [ ! -d "$input_folder" ]; then
        continue
    fi
    scene=$(basename $input_folder)

    # Skip specific scenes
    if [[ " ${skip_scenes_array[@]} " =~ " ${scene} " ]]; then
        echo "Skipping scene $scene"
        continue
    fi

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
        --experiment-name $scene \
        --project-name WheatGS-nerfstudio \
        --vis wandb \
        nerfstudio-data

    # Get directory
    latest_exp_name=$(ls $model_path | sort -r | head -n 1)
    exp_path="$model_path/$latest_exp_name"

    # Render and calculate metrics
    ns-render dataset --load_config "$exp_path/config.yml" --split train+test --output-path "$exp_path"
    ns-eval --load-config "$exp_path/config.yml" --output-path "$exp_path/$results_json"

done
echo "Done processing all scenes"

