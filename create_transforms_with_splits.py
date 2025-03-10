import os
import json
import re
import copy
import numpy as np
from pathlib import Path

def create_transforms_with_splits(transforms_json_path, output_suffix="_with_splits"):
    """
    Create a modified transforms.json that includes train, val, and test filename lists.
    
    Args:
        transforms_json_path (str): Path to the original transforms.json file
        output_suffix (str): Suffix to add to the output filename
    """
    # Load the transforms.json file
    with open(transforms_json_path, 'r') as f:
        transforms_data = json.load(f)
    
    # Create a copy for the modified data
    modified_data = copy.deepcopy(transforms_data)
    
    # Get all frames
    frames = transforms_data.get("frames", [])
    
    # Convert frame paths to Path objects for consistent parsing
    image_paths = [Path(frame["file_path"]) for frame in frames]
    
    # Extract camera IDs from filenames
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
    
    # Create indices for train, val, and test sets
    num_images = len(frames)
    i_all = np.arange(num_images)
    i_train = [i for i, v in enumerate(last_parts) if v <= 10]
    i_test = [i for i, v in enumerate(last_parts) if v > 10]
    
    # For this case, val and test are the same
    i_val = i_test.copy()
    
    # Verify all images are accounted for
    assert len(i_all) == len(i_train) + len(i_test)
    
    # Get filenames for each split
    train_filenames = [frames[i]["file_path"] for i in i_train]
    val_filenames = [frames[i]["file_path"] for i in i_val]
    test_filenames = [frames[i]["file_path"] for i in i_test]
    
    # Add filename lists to the modified data
    modified_data["train_filenames"] = train_filenames
    modified_data["val_filenames"] = val_filenames
    modified_data["test_filenames"] = test_filenames
    
    # Determine output path
    input_path = Path(transforms_json_path)
    output_filename = f"{input_path.stem}{output_suffix}{input_path.suffix}"
    output_path = str(input_path.parent / output_filename)
    
    # Save the modified json file
    with open(output_path, 'w') as f:
        json.dump(modified_data, f, indent=2)
    
    # Print statistics
    print(f"Total frames: {num_images}")
    print(f"Training frames: {len(train_filenames)}, camera IDs <= 10")
    print(f"Validation frames: {len(val_filenames)}, camera IDs > 10")
    print(f"Testing frames: {len(test_filenames)}, camera IDs > 10")
    
    print(f"\nModified transforms.json saved to: {output_path}")
    
    return {
        "output_path": output_path,
        "num_train": len(train_filenames),
        "num_val": len(val_filenames),
        "num_test": len(test_filenames)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create transforms.json with train/val/test splits")
    parser.add_argument("--transforms_json", required=True, help="Path to the transforms.json file")
    parser.add_argument("--output_suffix", default="_with_splits", help="Suffix for output filename")
    
    args = parser.parse_args()
    
    result = create_transforms_with_splits(args.transforms_json, args.output_suffix)
    
    print("\nUsage:")
    print(f"This file can be used with custom dataparsers that recognize the train_filenames, val_filenames, and test_filenames fields.")