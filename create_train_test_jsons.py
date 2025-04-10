import os
import json
import re
import copy
import numpy as np
from pathlib import Path

def create_train_test_splits(transforms_json_path, output_prefix=""):
    """
    Create separate train and test transforms.json files for Nerfstudio,
    placing images from cam_11 and cam_12 in test and all others in train.
    
    Args:
        transforms_json_path (str): Path to the original transforms.json file
        output_prefix (str): Prefix for output filenames (default: use same directory as input)
    """
    # Load the transforms.json file
    with open(transforms_json_path, 'r') as f:
        transforms_data = json.load(f)
    
    # Create copies for train and test
    train_data = copy.deepcopy(transforms_data)
    test_data = copy.deepcopy(transforms_data)
    
    # Clear frames lists
    train_data["frames"] = []
    test_data["frames"] = []
    
    # Get all frames
    frames = transforms_data.get("frames", [])
    
    # Convert frame paths to Path objects for consistent parsing
    image_paths = [Path(frame["file_path"]) for frame in frames]
    
    # Extract camera IDs from filenames
    # Assuming format like "FPWW036_SR0461_FIP2_cam_12.png" where the last part after "cam_" is the ID
    last_parts = []
    for path in image_paths:
        # Extract the number after "cam_"
        match = re.search(r'cam_(\d+)', path.name)
        if match:
            cam_id = int(match.group(1))
            last_parts.append(cam_id)
        else:
            # Default value if no camera ID found
            last_parts.append(0)
    
    # Create indices for train and test sets
    num_images = len(frames)
    i_all = np.arange(num_images)
    i_train = [i for i, v in enumerate(last_parts) if v <= 10]
    i_eval = [i for i, v in enumerate(last_parts) if v > 10]
    
    # Verify all images are accounted for
    assert len(i_all) == len(i_train) + len(i_eval)
    
    # Assign frames to train and test sets
    train_frames = [frames[i] for i in i_train]
    test_frames = [frames[i] for i in i_eval]
    
    train_data["frames"] = train_frames
    test_data["frames"] = test_frames
    
    # Determine output paths
    input_path = Path(transforms_json_path)
    if output_prefix:
        train_path = f"{output_prefix}_train.json"
        test_path = f"{output_prefix}_test.json"
        split_path = f"{output_prefix}_split.json"
    else:
        train_path = str(input_path.parent / "transforms_train.json")
        test_path = str(input_path.parent / "transforms_test.json")
        split_path = str(input_path.parent / "split.json")
    
    # Save the train and test json files
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Also create a split.json file for compatibility with --data.split-path
    split_data = {
        "train": [frame["file_path"] for frame in train_frames],
        "test": [frame["file_path"] for frame in test_frames]
    }
    
    with open(split_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    # Print statistics
    print(f"Total frames: {num_images}")
    print(f"Training frames: {len(train_frames)}, camera IDs <= 10")
    print(f"Testing frames: {len(test_frames)}, camera IDs > 10")
    
    print(f"\nTrain indices: {i_train}")
    print(f"Train filenames: {[Path(frames[i]['file_path']).name for i in i_train]}")
    
    print(f"\nTest indices: {i_eval}")
    print(f"Test filenames: {[Path(frames[i]['file_path']).name for i in i_eval]}")
    
    print(f"\nTrain JSON saved to: {train_path}")
    print(f"Test JSON saved to: {test_path}")
    print(f"Split JSON saved to: {split_path}")
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "split_path": split_path,
        "num_train": len(train_frames),
        "num_test": len(test_frames)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create train and test splits for Nerfstudio")
    parser.add_argument("--transforms_json", required=True, help="Path to the transforms.json file")
    parser.add_argument("--output_prefix", default="", help="Prefix for output filenames (default: use same directory as input)")
    
    args = parser.parse_args()
    
    result = create_train_test_splits(args.transforms_json, args.output_prefix)

    # Remove file extensions from paths, for nerfstudio blender data format
    os.system(f"sed -i 's/\.\(png\|jpg\)//g' {result['train_path']}")
    os.system(f"sed -i 's/\.\(png\|jpg\)//g' {result['test_path']}")
    os.system(f"cp {result['test_path']} {result['test_path'].replace('test.json', 'val.json')}")