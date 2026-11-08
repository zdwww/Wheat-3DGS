# Combining structural traits data of all plots
# (intended use after genotype assignment)
# Authors: Tomislav Medic and ChatGPT 28.02.2025

from functions.data_analysis_utils import *

from pathlib import Path


# === USER‐CONFIGURABLE PARAMETERS ===
DATA_RELATIVE_PATH = "results/genotypes_assigned/gs"
OUTPUT_RELATIVE_PATH = "results/traits_and_genotypes_all_methods"


def main(data_relative_path=DATA_RELATIVE_PATH, output_relative_path=OUTPUT_RELATIVE_PATH) -> None:

    # Manually adjustable parameters
    # Data-path
    project_folder = Path(__file__).parent
    data_path = project_folder / Path(data_relative_path)
    output_folder = Path(output_relative_path)
    output_file_path_and_name = output_relative_path + "/" + data_path.name

    output_folder.mkdir(parents=True, exist_ok=True)

    # Load all data (plots in .csv), concatenate and add plot_id
    data = load_and_concat_trait_lists(data_path)

    # Clean data (fill negative volumes and NaNs with 0)
    data = clean_traits_data(data)

    # Create unique_id per sample
    data = create_unique_id(data)

    # Sort data by 'unique_id' and reset the index
    data = data.sort_values(by='unique_id').reset_index(drop=True)

    data.to_csv(f"{output_file_path_and_name}_all.csv", index=False)
    return None


if __name__ == "__main__":
    main()




