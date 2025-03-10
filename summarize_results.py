import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def get_results_per_scene(base_path: Path, method: str, output_folder: Path):
    """
    Extract metrics for each scene (plot) with a specific NeRF method.
    
    Args:
        base_path: Path to the outputs directory containing all plots
        method: The method name (e.g., 'nerfacto')
        output_folder: Where to save the resulting CSVs
    """
    results = []
    
    # Find all plot folders (scenes)
    plot_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('plot_')]
    
    for plot_folder in sorted(plot_folders):
        scene_name = plot_folder.name
        method_folder = plot_folder / method
        
        if not method_folder.exists() or not method_folder.is_dir():
            print(f"Method {method} not found for {scene_name}")
            continue
            
        # Find the latest run (assuming format YYYY-MM-DD_HHMMSS)
        run_folders = [f for f in method_folder.iterdir() if f.is_dir() and len(f.name) == 17 and f.name[4] == '-']
        if not run_folders:
            print(f"No run folders found for {scene_name}/{method}")
            continue
            
        # Sort by date to get the latest run
        latest_run = sorted(run_folders, key=lambda x: datetime.strptime(x.name, "%Y-%m-%d_%H%M%S"))[-1]
        
        # Find test results
        test_results_path = latest_run / 'test_results.json'
        if not test_results_path.exists():
            print(f"test_results.json not found for {scene_name}/{method}/{latest_run.name}")
            continue
            
        # Load test results
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
            psnr = test_results["results"].get("psnr", "N/A")
            ssim = test_results["results"].get("ssim", "N/A")
            lpips = test_results["results"].get("lpips", "N/A")
            fps = test_results["results"].get("fps", "N/A")

        # Find checkpoint for storage calculation in nerfstudio_models
        model_checkpoint_path = latest_run / "nerfstudio_models"
        checkpoint_files = list(model_checkpoint_path.glob('*.ckpt')) if model_checkpoint_path.exists() else []
        
        if not checkpoint_files:
            print(f"No checkpoint found for {scene_name} in nerfstudio_models")
            storage_gb = "N/A"
        else:
            # Use the latest checkpoint if multiple exist
            checkpoint = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime)[-1]
            storage_gb = round(checkpoint.stat().st_size / 1024**3, 3)
            iters = int(checkpoint.stem.split('-')[-1]) + 1

        # Find wandb metrics
        wandb_run_path = latest_run / 'wandb' / 'latest-run' / 'files'
        summary_json_path = wandb_run_path / 'wandb-summary.json'
        metadata_json_path = wandb_run_path / 'wandb-metadata.json'
        
        # Default values
        runtime = "N/A"
        total_gaussians = "N/A"
        gpu_model = "N/A"
        
        # Get summary data
        if summary_json_path.exists():
            with open(summary_json_path, 'r') as f:
                wandb_data = json.load(f)
                runtime = int(wandb_data.get('_runtime', 0) / 60)
                total_gaussians = wandb_data.get('total_points', 'N/A')
        else:
            print(f"Wandb summary not found for {scene_name}/{method}/{latest_run.name}")
        
        # Get GPU model from metadata
        if metadata_json_path.exists():
            with open(metadata_json_path, 'r') as f:
                metadata = json.load(f)
                gpu_model = metadata.get('gpu', 'N/A')
        else:
            print(f"Wandb metadata not found for {scene_name}/{method}/{latest_run.name}")

        # Collect all metrics for this scene
        metrics = {
            'Scene': scene_name,
            'Method': method,
            'Run': latest_run.name,
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips,
            'FPS': fps,
            'Runtime (min)': runtime,
            'Total Gaussians': total_gaussians,
            'Storage (GB)': storage_gb,
            'GPU Model': gpu_model,
        }
        results.append(metrics)

    if not results:
        print(f"No results found for method: {method}")
        return

    # Create DataFrame from scene metrics
    df_results = pd.DataFrame(results)

    # Save per-scene metrics to CSV
    output_folder.mkdir(parents=True, exist_ok=True)
    per_scene_csv_path = output_folder / f'{method}_per_scene_metrics.csv'
    df_results.to_csv(per_scene_csv_path, index=False)
    print(f"Per-scene metrics saved to: {per_scene_csv_path}")

    # Calculate averages and standard deviations
    numeric_keys = [
        'PSNR', 'SSIM', 'LPIPS', 'FPS',
        'Runtime (min)', 'Total Gaussians', 'Storage (GB)'
    ]
    avg_metrics = {'Metric': 'Average', 'Method': method}
    std_metrics = {'Metric': 'Std Dev', 'Method': method}

    for key in numeric_keys:
        # Collect valid numeric values
        values = [float(r[key]) for r in results if r[key] != 'N/A' and str(r[key]).replace('.', '', 1).isdigit()]

        if values:  # Only calculate if there are valid values
            avg_metrics[key] = round(np.mean(values), 3)
            std_metrics[key] = round(np.std(values), 3)
        else:
            avg_metrics[key] = 'N/A'
            std_metrics[key] = 'N/A'

    # Create DataFrame for summary statistics
    df_summary = pd.DataFrame([avg_metrics, std_metrics])

    # Save summary statistics to CSV
    summary_csv_path = output_folder / f'{method}_summary_metrics.csv'
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary metrics saved to: {summary_csv_path}")
    
    return df_results

def compare_methods(base_path: Path, nerfstudio_models_path: Path, methods: list, output_folder: Path):
    """
    Compare metrics across different methods.
    
    Args:
        base_path: Path to the outputs directory containing all plots
        nerfstudio_models_path: Path to nerfstudio_models directory with checkpoint files
        methods: List of method names to compare
        output_folder: Where to save the resulting CSVs
    """
    all_summaries = []
    all_results = []
    
    for method in methods:
        print(f"\nProcessing method: {method}")
        df_results = get_results_per_scene(base_path, nerfstudio_models_path, method, output_folder)
        
        if df_results is not None:
            all_results.append(df_results)
            
            # Read summary file we just created
            summary_csv_path = output_folder / f'{method}_summary_metrics.csv'
            if summary_csv_path.exists():
                df_summary = pd.read_csv(summary_csv_path)
                all_summaries.append(df_summary)
    
    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results)
        combined_csv_path = output_folder / 'all_methods_per_scene.csv'
        combined_results.to_csv(combined_csv_path, index=False)
        print(f"\nCombined per-scene metrics saved to: {combined_csv_path}")
    
    if all_summaries:
        # Combine all summaries
        combined_summary = pd.concat(all_summaries)
        combined_summary_path = output_folder / 'all_methods_summary.csv'
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"Combined summary metrics saved to: {combined_summary_path}")


# Usage Example
if __name__ == "__main__":
    base_path = Path('outputs')
    output_folder = Path('results')
    
    # Single method analysis
    method = "nerfacto"
    get_results_per_scene(base_path, method, output_folder / method)
    
    # Or compare multiple methods
    # methods = ["nerfacto", "instant-ngp", "gaussian-splatting"]
    # compare_methods(base_path, methods, output_folder)