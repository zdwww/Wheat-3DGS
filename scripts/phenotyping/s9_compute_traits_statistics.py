#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from functions.data_analysis_utils import *

# === USER‐CONFIGURABLE PARAMETERS ===
# Define the folder path manually using pathlib
DATA_RELATIVE_PATH = Path("results/traits_and_genotypes_all_methods")
OUTPUT_RELATIVE_PATH = Path("results/statistics_tables_3_and_4")
# List of traits of interest
TRAITS_TO_ANALYZE = ['length', 'width', 'volume']
# Outlier removal before statistics computation
APPLY_OUTLIER_REMOVAL = True
OUTLIER_THRESHOLD = 3


def main(data_relative_path=DATA_RELATIVE_PATH, output_relative_path=OUTPUT_RELATIVE_PATH,
         traits_to_analyze=TRAITS_TO_ANALYZE, apply_outlier_removal=APPLY_OUTLIER_REMOVAL,
         outlier_threshold=OUTLIER_THRESHOLD) -> None:

    # Set paths:
    data_path = Path(data_relative_path)
    results_path = Path(output_relative_path)

    # Run the simple statistical analysis for all samples per method in a bulk (mean, median, std, scaled mad):
    compute_basic_statistics(data_path, traits_to_analyze, results_path, apply_outlier_removal, outlier_threshold)

    # Run the regression analysis (per sample analysis):
    results_reg_instance = run_regression_analysis(data_path, traits_to_analyze, results_path,
                                                   apply_outlier_removal=apply_outlier_removal,
                                                   outlier_threshold=outlier_threshold)

    # Compute Cohen's f, trait stability, (ANOVA) F-test with corresponding P-value
    # - checking if the derived traits can be useful for genotyping (group-wise phenotyping)
    results_disc_analysis = trait_discriminativeness_analysis(data_path, traits_to_analyze,
                                                              results_path, apply_outlier_removal=apply_outlier_removal,
                                                              outlier_threshold=outlier_threshold)

    # Compare per-plot average values against reference technologies
    results_reg_row = run_per_row_regression_analysis(data_path, traits_to_analyze, results_path,
                                                      apply_outlier_removal=apply_outlier_removal,
                                                      outlier_threshold=outlier_threshold, agg_method="median")

    # Compute and save table 3:
    compute_table_3(traits_to_analyze, results_reg_instance, results_reg_row, results_path)

    # Compute and save table 4:
    compute_table_4(traits_to_analyze, results_disc_analysis, results_path)
    return None


if __name__ == "__main__":
    main()




