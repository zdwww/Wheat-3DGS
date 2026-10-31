import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.path import Path as MplPath
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from itertools import chain

# --- Helper Functions ---
def is_point_in_polygon(point: np.ndarray, poly_xy: np.ndarray) -> bool:
    """
    Checks if a 2D point is inside a polygon defined by poly_xy (an array of shape (n_points, 2)).
    Uses matplotlib.path.Path.contains_point.
    """
    path = MplPath(poly_xy)
    return path.contains_point(point)


def create_scatter_plot_random_colors(df, field_name: str = 'gen_row_id', seed=None):
    """
    Creates a scatter plot using the 'X' and 'Y' columns of df, where each point is colored
    based on its 'bbox_id'. Each unique bbox_id is assigned a quasi-random color.
    Rows with NaN in 'bbox_id' are assigned a default color.

    Parameters:
        df (pandas.DataFrame): DataFrame containing at least the columns 'X', 'Y', and 'bbox_id'.
        seed (int, optional): Seed for the random number generator (for reproducibility).
        :param field_name: field name we would like to color

    Returns:
        None. The plot is displayed.

    """
    # Optionally set the random seed for reproducibility.
    if seed is not None:
        np.random.seed(seed)

    # Define a default color for NaN values in bbox_id (e.g., gray)
    nan_color = (0.5, 0.5, 0.5)

    # Get unique bbox_ids that are not NaN.
    unique_ids = df[field_name].dropna().unique()

    # Create a dictionary mapping each unique bbox_id to a random RGB color.
    random_colors = {bbox_id: tuple(np.random.rand(3).tolist()) for bbox_id in unique_ids}

    # For each row, assign its color based on bbox_id.
    # If bbox_id is NaN, assign the default nan_color.
    colors = [random_colors[bbox] if pd.notna(bbox) else nan_color for bbox in df[field_name]]

    # Create the scatter plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['X'], df['Y'], color=colors, s=50)

    # Build legend handles for non-NaN bbox_ids.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="None", markersize=8,
                   color=random_colors[bbox], label=f'gen_{bbox}')
        for bbox in unique_ids
    ]

    # Optionally, add a legend entry for NaN if any exist.
    if df[field_name].isna().any():
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="None", markersize=8,
                       color=nan_color, label='gen_id NaN')
        )

    # ax.legend(handles=handles, title="bbox_id", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set axis labels and title.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatter Plot with Quasi-Random Colors by gen_id')

    # Adjust layout and display the plot.
    plt.tight_layout()
    plt.show()


def plot_polygon_and_traits(poly_xy, traits_xy):
    """
    Plots a 2D scatter with two sets of points:
      - Blue markers for the polygon points.
      - Red markers for the trait points.

    Parameters:
        poly_xy (np.ndarray): Array of shape (n_points, 2) containing polygon vertex coordinates.
        traits_xy (np.ndarray): Array of shape (m_points, 2) containing trait point coordinates.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(poly_xy[:, 0], poly_xy[:, 1], color='blue', label='Polygon Points')
    plt.scatter(traits_xy[:, 0], traits_xy[:, 1], color='red', label='Trait Points')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygon vs. Trait Points")
    plt.legend()
    plt.show()