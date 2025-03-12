# Default values
dataset_folder="/workspace/Wheat-GS/dataset/Wheat-GS-data-nerf/20240717"
transforms_json=transforms_with_splits.json
sparse_pc=sparse_pc.ply

# Iterate through all input folders
for input_folder in "$dataset_folder"/*; do
    if [ ! -d "$input_folder" ]; then
        continue
    fi
    scene=$(basename $input_folder)
    nerfacto_folder="$input_folder/nerfstudio"
    
    if [ ! -d "$nerfacto_folder" ] || [ ! -f "$nerfacto_folder/$transforms_json" ] || [ ! -f "$nerfacto_folder/$sparse_pc" ]; then
        echo "Preprocessing scene $scene"
        
        # Convert COLMAP model
        colmap model_converter --input_path "$input_folder/sparse/0/" --output_path "$input_folder/sparse/0/" --output_type BIN
        
        # Process data with ns-process-data
        ns-process-data images --data "$input_folder" --skip-colmap --skip-image-processing --output-dir "$nerfacto_folder" --colmap-model-path "../sparse/0"
        
        # Create transforms with splits
        python create_transforms_with_splits.py --transforms_json $nerfacto_folder/transforms.json --output_suffix _with_splits
    fi
done
echo "Done preprocessing all scenes"
