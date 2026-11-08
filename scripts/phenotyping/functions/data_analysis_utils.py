import pathlib
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from math import sqrt


def clean_traits_data(traits: pd.DataFrame, winsorize_percent: float = 1.0) -> pd.DataFrame:
    """
    Clean values for bad samples in a pandas dataframe containing all traits for all samples
    """

    # Remove negative volumes
    for col in traits.columns:
        if 'vol' in col.lower():
            traits.loc[traits[col] < 0, col] = 0
            traits[col] *= 10**6

    # Replace NaNs
    traits.fillna(0, inplace=True)

    return traits


def create_unique_id(data: pd.DataFrame) -> pd.DataFrame:
    data['unique_id'] = data['file_id'].astype(str) + "_" + data['plot_id'].astype(str)
    return data


def load_and_concat_trait_lists(folder):
    file_list = list(folder.glob("*.xlsx")) + list(folder.glob("*.csv"))
    df_list = []
    for file in file_list:
        if file.suffix == '.csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        # Extract the first 3 characters from the file name (stem) as plot_id
        df['plot_id'] = file.stem[:3]
        # (Assuming the file already contains a "file_id" column)
        df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()


def remove_outliers(series, threshold=3):
    """
    Remove outliers from a pandas Series based on a threshold times the scaled median absolute deviation (MAD).

    Parameters:
      series (pd.Series): Input data.
      threshold (float): Number of scaled MADs from the median to keep.

    Returns:
      pd.Series: Filtered series.
    """
    series = series.dropna()
    median_val = series.median()
    # Compute the scaled MAD (using factor 1.4826)
    scaled_mad = (np.abs(series - median_val)).median() * 1.4826
    lower_bound = median_val - threshold * scaled_mad
    upper_bound = median_val + threshold * scaled_mad
    return series[(series >= lower_bound) & (series <= upper_bound)]


def compute_basic_statistics(data_path: Path, traits_to_analyze: list, results_path: Path,
                             apply_outlier_removal: bool = True, outlier_threshold: Union[float, int] = 3) -> None:
    # Loop over CSV files and compute statistics (mean, median, std, scaled mad)
    # Dictionaries to hold statistics for each file for all rows and for filtered rows
    stats_all = {}
    found_traits = []
    # Make directory for storing results (if necessary)
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all CSV files in the folder
    for csv_file in data_path.glob("*.csv"):

        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Dictionaries to store statistics for the current file
        file_stats_all = {}

        # Loop over columns in the current file
        for col in df.columns:
            # Check if the column name contains any of the trait substrings (case-insensitive)
            if any(trait in col.lower() for trait in traits_to_analyze):
                if col not in found_traits:
                    found_traits.append(col)

                # ---- For all samples ----
                series_all = df[col].dropna()

                if apply_outlier_removal:
                    series_all = remove_outliers(series_all, threshold=outlier_threshold)

                mean_val = series_all.mean()
                median_val = series_all.median()
                std_val = series_all.std()
                scaled_mad = (np.abs(series_all - median_val)).median() * 1.4826

                file_stats_all[col] = {
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    "scaled_mad": scaled_mad
                }

        # Use the file stem (e.g. "data1", "data2", etc.) as the key
        file_key = csv_file.stem
        stats_all[file_key] = file_stats_all

    # ---------------------------
    # 4. Aggregate and compare statistics across files
    # ---------------------------
    # We create  Excel workbook with one sheet per trait; in each sheet, each column corresponds to a file (instrument)
    # and each row corresponds to one of the computed statistics (mean, median, std, scaled mad).

    def compile_trait_data(stats_dict, trait):
        """
        For a given stats dictionary (either for all or filtered samples) and a trait string,
        returns a DataFrame with rows as statistics and columns as file identifiers.
        """
        trait_data = {}
        for file_key, file_stats in stats_dict.items():
            # Look for the first column in the file whose name contains the trait substring
            for col, stat in file_stats.items():
                if trait in col.lower():
                    trait_data[file_key] = stat
                    break  # if multiple columns match, only the first encountered is used
        return pd.DataFrame(trait_data, index=["mean", "median", "std", "scaled_mad"])

    # --- Create Excel workbook for all samples ---
    # all_excel_path = results_path / "traits_stats_comparison_all_instruments.xlsx"
    #
    # with pd.ExcelWriter(all_excel_path) as writer_all:
    #     for trait in found_traits:
    #         df_trait_all = compile_trait_data(stats_all, trait)
    #         df_trait_all.to_excel(writer_all, sheet_name=trait)
    #
    # print("Analysis complete.")
    # print(f"Excel workbook for all samples saved to: {all_excel_path}")


def compute_regression_metrics(x, y):
    """
    Given two arrays (x - reference and y - query), compute regression metrics:
      - Pearson correlation coefficient and p-value,
      - R^2 (square of the Pearson correlation),
      - RMSE,
      - MAE.
    Returns a dictionary with these metrics.
    """
    if len(x) < 2 or len(y) < 2:
        return {'corr': np.nan, 'p_value': np.nan, 'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}

    corr, p_value = pearsonr(x, y)
    r2 = corr ** 2
    rmse = sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    mape = np.mean(np.abs((x - y) / x)) * 100
    return {'corr': corr, 'p_value': p_value, 'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}


def run_regression_analysis(data_path, traits_to_regress, results_path,
                            apply_outlier_removal=False, outlier_threshold=3) -> dict:
    """
    Identify the TLS file (the one with "tls" in its name) as ground truth and treat all other CSV files
    (excluding ones with "sls" or "labels" in their names) as query files.

    For each trait in traits_to_regress, extract the column from the TLS file and from each query file
    (by matching on the 'unique_id' field). Optionally remove outliers if apply_outlier_removal==True.

    For each (tls, query) pair compute:
      - Pearson correlation coefficient and p-value,
      - R² (as the square of the correlation),
      - RMSE and MAE.

    The metrics are computed twice: once for all samples and once for the subset where the corresponding
    'unique_id' in good_bad_labels has a label value of 0.

    The results are saved as two Excel files (one for all samples and one for filtered samples) with one
    sheet per trait. In each sheet, the columns correspond to the query file (by file stem) and the rows
    are the computed metrics.
    """
    # Identify TLS file and query files
    # results_path = results_path / "per_sample_regression"
    # results_path.mkdir(parents=True, exist_ok=True)
    tls_file = None
    query_files = []
    for csv_file in data_path.glob("*.csv"):
        name_lower = csv_file.name.lower()
        if "tls" in name_lower:
            tls_file = csv_file
        else:
            query_files.append(csv_file)
    if tls_file is None:
        print("No TLS file found in", data_path)
        return

    # Load TLS file
    df_tls = pd.read_csv(tls_file)

    # For each trait, determine the column in the TLS file
    tls_trait_cols = {}
    for trait in traits_to_regress:
        found = None
        for col in df_tls.columns:
            if trait.lower() in col.lower():
                found = col
                break
        if found is not None:
            tls_trait_cols[trait] = found
        else:
            print(f"Trait '{trait}' not found in TLS file {tls_file.name}.")

    # Prepare dictionaries to collect metrics for each trait (for all and filtered samples)
    results_all = {trait: {} for trait in traits_to_regress}

    # Process each query file
    for query_file in query_files:
        df_query = pd.read_csv(query_file)
        for trait in traits_to_regress:
            # Find matching column in query file
            query_col = None
            for col in df_query.columns:
                if trait.lower() in col.lower():
                    query_col = col
                    break
            if query_col is None or trait not in tls_trait_cols:
                continue  # Skip if the trait is not present in one of the files

            # Merge on unique_id to align TLS and query values
            merged_all = pd.merge(df_tls[['unique_id', tls_trait_cols[trait]]],
                                  df_query[['unique_id', query_col]],
                                  on='unique_id', how='inner')

            merged_all = merged_all.rename(columns={f"{tls_trait_cols[trait]}_x": 'tls', f"{query_col}_y": 'query'})

            # Optionally remove outliers
            if apply_outlier_removal:

                merged_all['tls'] = remove_outliers(merged_all['tls'], threshold=outlier_threshold)
                merged_all['query'] = remove_outliers(merged_all['query'], threshold=outlier_threshold)
                merged_all = merged_all.dropna(subset=['tls', 'query'])

            # Compute metrics
            metrics_all = compute_regression_metrics(merged_all['tls'].values, merged_all['query'].values)

            # Use query file stem as identifier
            results_all[trait][query_file.stem] = metrics_all

    # Write the results to Excel workbooks
    # excel_all = results_path / "regression_metrics_all.xlsx"
    #
    # with pd.ExcelWriter(excel_all) as writer_all:
    #     for trait in traits_to_regress:
    #         df_trait = pd.DataFrame(results_all[trait])
    #         df_trait = df_trait.reindex(['corr', 'p_value', 'r2', 'rmse', 'mae', 'mape'])
    #         df_trait.to_excel(writer_all, sheet_name=trait[:31])
    #
    # print("Regression analysis complete.")
    # print("All samples results saved to:", excel_all)
    return results_all


def compute_anova_stats(values, groups):
    """
    Computes Cohen's f, Trait Stability, F-ratio, and p-value from a one-way ANOVA.

    Parameters:
        values (np.ndarray): 1D array of trait values.
        groups (np.ndarray): 1D array of group labels (e.g., gen_row_id) corresponding to each value.

    Returns:
        cohen_f (float): Effect size measure. Higher values indicate stronger discrimination.
        trait_stability (float): Ratio of average within-group variance to overall variance.
                                 Lower values indicate that most variation is between groups.
        f_ratio (float): F-statistic from one-way ANOVA.
        p_value (float): p-value corresponding to the F-ratio.
    """
    unique_groups = np.unique(groups)
    group_samples = []
    group_variances = []

    overall_mean = np.mean(values)
    ss_total = np.sum((values - overall_mean) ** 2)

    ss_between = 0
    for group in unique_groups:
        group_data = values[groups == group]
        n_group = len(group_data)
        if n_group < 3:
            continue
        group_samples.append(group_data)
        group_mean = np.mean(group_data)
        ss_between += n_group * (group_mean - overall_mean) ** 2
        # Compute the within-group variance (using ddof=1 for an unbiased estimator)
        group_variances.append(np.var(group_data, ddof=1))

    # Eta squared: proportion of variance explained by group differences
    eta_squared = ss_between / ss_total if ss_total != 0 else np.nan
    # Cohen's f: effect size (avoid division by zero)
    cohen_f = np.sqrt(eta_squared / (1 - eta_squared)) if (1 - eta_squared) != 0 else np.nan

    # Trait stability: average within-group variance divided by overall variance
    avg_within_variance = np.mean(group_variances)
    overall_variance = np.var(values, ddof=1)
    trait_stability = avg_within_variance / overall_variance if overall_variance != 0 else np.nan

    # Perform one-way ANOVA to obtain F-ratio and p-value
    try:
        f_ratio, p_value = stats.f_oneway(*group_samples)
    except Exception:
        f_ratio, p_value = np.nan, np.nan

    return cohen_f, trait_stability, f_ratio, p_value


def trait_discriminativeness_analysis(data_path, traits_to_regress, results_path,
                        apply_outlier_removal=True, outlier_threshold=3) -> dict:

    # Dictionaries to store results
    results_all = {trait: {} for trait in traits_to_regress}

    # ---------------------------
    # 1. Loop over CSV files (excluding labels.csv)
    # ---------------------------
    for csv_file in data_path.glob("*.csv"):

        df = pd.read_csv(csv_file)
        file_key = csv_file.stem  # use file stem as identifier
        df.name = file_key

        # --- Analysis for all samples ---
        for trait in traits_to_regress:
            if trait in df.columns and 'gen_row_id' in df.columns:
                series_all = df[trait].dropna()
                if apply_outlier_removal:
                    series_all = remove_outliers(series_all, threshold=outlier_threshold)
                # Align groups with the filtered indices
                filtered_df_all = df.loc[series_all.index]
                groups_all = filtered_df_all['gen_row_id'].values
                values_all = series_all.values

                # Compute the metrics
                cohen_f, trait_stability, f_ratio, p_value = compute_anova_stats(values_all, groups_all)
                results_all[trait][file_key] = {
                    "Cohen's f": cohen_f,
                    "Trait Stability": trait_stability,
                    "F-ratio": f_ratio,
                    "p-value": p_value
                }

    # ---------------------------
    # 2. Save the results to Excel workbooks
    # ---------------------------
    # For all samples
    # output_all = results_path / "anova_statistics_all.xlsx"
    # with pd.ExcelWriter(output_all) as writer_all:
    #     for trait, file_results in results_all.items():
    #         # Create a DataFrame with rows as statistics and columns as file names
    #         df_trait = pd.DataFrame(file_results).T  # files as rows, metrics as columns
    #         df_trait = df_trait.T  # transpose so that rows correspond to metrics
    #         df_trait.to_excel(writer_all, sheet_name=trait)
    #
    # print("ANOVA statistics computed and saved:")
    # print(f" - All samples: {output_all}")
    return results_all


def run_per_row_regression_analysis(data_path, traits_to_regress, results_path,
                                    apply_outlier_removal=False, outlier_threshold=3,
                                    agg_method="median") -> dict:
    """
    For each CSV file in data_path, this function aggregates samples by 'gen_row_id'
    (only keeping groups with at least 3 samples) using either the mean or median (specified by agg_method).
    Outliers are removed (using remove_outliers()) both before aggregation and on the aggregated values if requested.

    Then for each query file (i.e. those that do not have "tls" in their name),
    the aggregated values are merged with the corresponding aggregated values from the TLS and SLS files based on "gen_row_id".

    For each trait in traits_to_regress, regression metrics (Pearson correlation, p-value, R², RMSE, MAE)
    are computed between the query file and the TLS (or SLS) aggregated samples.

    The results are saved to an Excel workbook:
      - "per_plot_regression_metrics_tls_all.xlsx" (all samples, compared against TLS)

    Parameters:
      data_path (Path): Directory containing the CSV files.
      traits_to_regress (list): List of trait substrings to be analyzed.
      results_path (Path): Directory where result files will be saved.
      apply_outlier_removal (bool): If True, remove outliers before computing aggregated values and metrics.
      outlier_threshold (float): Threshold multiplier for outlier removal (scaled MAD).
      agg_method (str): Aggregation method to use per "gen_row_id"; options are "median" or "mean".
    """

    # results_path = results_path / "per_plot_regression_statistics"
    # results_path.mkdir(parents=True, exist_ok=True)

    # Helper: Aggregate a dataframe by 'gen_row_id' for the given traits.
    def aggregate_by_bbox(df, traits, agg_method="median"):
        if 'gen_row_id' not in df.columns:
            print(f"File {df} has no 'gen_row_id' column. Skipping aggregation.")
            return pd.DataFrame()
        aggregated = []
        # Group by gen_row_id and only keep groups with at least 3 samples.
        for bbox, group in df.groupby('gen_row_id'):
            if len(group) < 3:
                continue
            agg_dict = {'gen_row_id': bbox}
            for trait in traits:
                # Find first column matching the trait substring (case-insensitive)
                trait_col = None
                for col in group.columns:
                    if trait.lower() in col.lower():
                        trait_col = col
                        break
                if trait_col is not None:
                    series = group[trait_col].dropna()
                    if apply_outlier_removal:
                        series = remove_outliers(series, threshold=outlier_threshold)
                    if series.empty:
                        agg_val = np.nan
                    else:
                        if agg_method == "median":
                            agg_val = series.median()
                        else:
                            agg_val = series.mean()
                    agg_dict[trait] = agg_val
            aggregated.append(agg_dict)
        agg_df = pd.DataFrame(aggregated)
        # Optionally remove outliers from the aggregated values
        if apply_outlier_removal:
            for trait in traits:
                if trait in agg_df.columns:
                    agg_df[trait] = remove_outliers(agg_df[trait], threshold=outlier_threshold)
        return agg_df

    # Identify TLS and query files
    tls_file = None,
    query_files = []
    for csv_file in data_path.glob("*.csv"):
        name_lower = csv_file.name.lower()
        if "tls" in name_lower:
            tls_file = csv_file
        else:
            query_files.append(csv_file)

    if tls_file is None:
        print("TLS file found. Exiting regression analysis.")
        return

    # Read and aggregate TLS and SLS files (for all samples and filtered samples)
    agg_tls = None
    if tls_file is not None:
        df_tls = pd.read_csv(tls_file)
        agg_tls = aggregate_by_bbox(df_tls, traits_to_regress, agg_method=agg_method)

    # Prepare dictionaries to store regression metrics for each trait
    results_tls_all = {trait: {} for trait in traits_to_regress}

    # Loop over query files to aggregate and merge with TLS data
    for query_file in query_files:
        df_query = pd.read_csv(query_file)
        agg_query = aggregate_by_bbox(df_query, traits_to_regress, agg_method=agg_method)
        if agg_query.empty:
            continue

        # Compare against TLS (all samples)
        if agg_tls is not None and not agg_tls.empty:
            merged_tls = pd.merge(agg_tls[['gen_row_id'] + traits_to_regress],
                                  agg_query[['gen_row_id'] + traits_to_regress],
                                  on='gen_row_id', suffixes=('_tls', '_query'))
            for trait in traits_to_regress:
                col_tls = f"{trait}_tls"
                col_query = f"{trait}_query"
                if col_tls in merged_tls.columns and col_query in merged_tls.columns:
                    merged_all = merged_tls.dropna(subset=[col_tls, col_query])
                    metrics = compute_regression_metrics(merged_all[col_tls].values, merged_all[col_query].values)
                    results_tls_all[trait][query_file.stem] = metrics

    # Write TLS comparison results to Excel workbooks
    # results_path.mkdir(parents=True, exist_ok=True)
    # excel_tls_all = results_path / "per_plot_regression_metrics_tls_all.xlsx"
    # with pd.ExcelWriter(excel_tls_all) as writer:
    #     for trait in traits_to_regress:
    #         df_trait = pd.DataFrame(results_tls_all[trait])
    #         df_trait = df_trait.reindex(['corr', 'p_value', 'r2', 'rmse', 'mae', 'mape'])
    #         df_trait.to_excel(writer, sheet_name=trait[:31])
    #
    # print("Per-plot regression analysis complete.")
    # print("TLS comparisons (all samples) saved to:", excel_tls_all)
    return results_tls_all


def compute_table_4(traits_to_analyze: list, results_disc_analysis: dict, results_path: Path) -> None:
    # Get methods/instruments to analyze
    methods_to_analyze = list(results_disc_analysis[traits_to_analyze[0]].keys())
    # Build a pandas MultiIndex for columns: first level = trait, second level = method
    columns = pd.MultiIndex.from_product(
        [traits_to_analyze, methods_to_analyze],
        names=("Trait", "Method"))

    # Extract “F-ratio” for each (trait, method) pair in row-major order
    row_values = [
        results_disc_analysis[trait][method]["F-ratio"]
        for trait in traits_to_analyze
        for method in methods_to_analyze
    ]
    df = pd.DataFrame([row_values], columns=columns)

    # Flatten the MultiIndex into single‐level column names (e.g. "length_tls_all", "length_gs_all", …)
    flat_columns = [f"{trait}_{method}" for (trait, method) in df.columns]
    df_flat = df.copy()
    df_flat.columns = flat_columns

    # Write out CSV and XLSX
    output_csv = results_path / "Table_4.csv"
    df_flat.to_csv(output_csv, index=False)
    print(f"Saved Table 4 → {output_csv}")
    return None


def compute_table_3(traits_to_analyze: list, results_reg_instance: dict, results_reg_row: dict,
                    results_path: Path) -> None:

    # Metrics for the table:
    metrics_to_analyze = ["corr", "mae", "mape"]

    # Get methods/instruments to analyze
    first_trait = traits_to_analyze[0]
    methods_to_analyze = list(results_reg_instance[first_trait].keys())

    # Build a row‐indexed DataFrame with MultiIndex on rows: (metric, method)
    # and MultiIndex on columns: (level, trait), where level is “per-instance” vs. “per-row”.

    row_index = pd.MultiIndex.from_product(
        [metrics_to_analyze, methods_to_analyze],
        names=("Metric", "Method")
    )

    col_index = pd.MultiIndex.from_product(
        [["per-instance", "per-row"], traits_to_analyze],
        names=("Level", "Trait")
    )

    # Prepare a list of lists: each sublist is one row in the final DataFrame
    all_rows = []
    for metric in metrics_to_analyze:
        for method in methods_to_analyze:
            # per‐instance values for each trait:
            per_instance_vals = [
                results_reg_instance[trait][method][metric]
                for trait in traits_to_analyze
            ]
            # per‐row‐average values for each trait:
            per_row_vals = [
                results_reg_row[trait][method][metric]
                for trait in traits_to_analyze
            ]
            # concatenate them in the correct column order:
            row_vector = per_instance_vals + per_row_vals
            all_rows.append(row_vector)

    # Create the DataFrame
    df = pd.DataFrame(all_rows, index=row_index, columns=col_index)

    # Reset the row index so Metric/Method become actual columns:
    df_reset = df.reset_index()

    # Flatten the MultiIndex column names for the per‐instance and per‐row levels:
    flat_column_names = []
    for lvl0, lvl1 in df_reset.columns:
        if isinstance(lvl0, str) and isinstance(lvl1, str):
            flat_column_names.append(f"{lvl0}_{lvl1}")
        else:
            name = lvl0 if isinstance(lvl0, str) else lvl1
            flat_column_names.append(name)

    df_flat = df_reset.copy()
    df_flat.columns = flat_column_names

    output_csv = results_path / "Table_3.csv"
    df_flat.to_csv(output_csv, index=False)
    print(f"Saved Table 3 → {output_csv}")
    return None
