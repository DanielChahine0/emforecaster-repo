#!/usr/bin/env python3
# -*- coding: utf‑8 -*-

"""
Heatmap Generator for EMF Data Analysis
Generates correlation heatmaps for:
1. Frequency bands (per-dataset actual frequencies)
2. Technology aggregations (2G/3G/4G/5G)
3. Operator aggregations (Iliad/TIM/VF/W3)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

def create_single_heatmap(title, data, output_folder, filename, trim_prefix=None):
    """Create correlation heatmap for given data, saved as filename in output_folder."""
    if data.empty or len(data.columns) < 2:
        print(f"   ⚠️  Skipping {title}: insufficient data ({len(data.columns)} columns)")
        return
    
    # Calculate correlation matrix
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='viridis', vmin=0, vmax=1)

    # Prepare tick labels (trim prefix if needed) with ENLARGED fonts
    xticklabels = list(corr.columns)
    yticklabels = list(corr.index)
    if trim_prefix:
        xticklabels = [str(c).replace(trim_prefix, "") for c in xticklabels]
        yticklabels = [str(c).replace(trim_prefix, "") for c in yticklabels]
        tick_fontsize = 15#14
        label_fontsize = 14
    else:
        # For frequency columns, use dynamic font size based on crowding
        if len(xticklabels) > 20:  # High density
            tick_fontsize = 15#6
            label_fontsize = 15#6
        elif len(xticklabels) > 15:  # Medium density
            tick_fontsize = 15#7
            label_fontsize = 15#7
        else:  # Low density
            tick_fontsize = 15#10
            label_fontsize = 15#10

    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=label_fontsize)
    ax.set_yticklabels(yticklabels, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Add correlation values as text - use white text for visibility on viridis
    if len(corr.columns) <= 10:  # Only add text for smaller matrices
        fontsize = 9 if trim_prefix else 6
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                        ha="center", va="center", fontsize=fontsize, color='white', fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Correlation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save heatmap
    output_path_png = output_folder / f"{filename}.png"
    output_path_pdf = output_folder / f"{filename}.pdf"

    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path_png.name}")
    plt.close(fig)

def get_actual_frequency_columns(df):
    """Get actual frequency columns from the dataset (numeric column names)."""
    freq_cols = []
    for col in df.columns:
        if isinstance(col, (int, float)):
            freq_cols.append(col)
        elif isinstance(col, str):
            try:
                # Check if string represents a frequency
                float_val = float(col)
                if 700 <= float_val <= 4000:  # Reasonable frequency range
                    freq_cols.append(col)
            except ValueError:
                continue
    
    # Sort frequencies properly (convert to float for sorting, then back to original type)
    try:
        freq_cols.sort(key=lambda x: float(x))
    except:
        freq_cols.sort()
    
    return freq_cols

def generate_dataset_heatmaps(df, dataset_name, dataset_folder):
    """Generate the per-dataset heatmaps into dataset_folder."""
    dataset_folder.mkdir(parents=True, exist_ok=True)

    print(f"   Generating heatmaps for {dataset_name}...")

    # 0. Combined trio heatmap (Frequency | Technology | Operators).
    create_dataset_trio_heatmap(df, dataset_name, dataset_folder)

    # 1. Frequency correlation heatmap (using actual frequencies in dataset).
    freq_cols = get_actual_frequency_columns(df)
    if len(freq_cols) >= 2:
        freq_data = df[freq_cols].ffill().bfill()
        if not freq_data.empty and freq_data.notna().sum().sum() > 0:
            create_single_heatmap(f"{dataset_name} - Frequency Bands", freq_data,
                                  dataset_folder, "Frequency bands")
        else:
            print(f"   No valid frequency data for {dataset_name}")
    else:
        print(f"   Insufficient frequency columns for {dataset_name} ({len(freq_cols)} found)")

    # 2. Technology aggregation heatmap.
    tech_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                 and any(tech in col for tech in ['2G', '3G', '4G', '5G'])]
    if len(tech_cols) >= 2:
        tech_data = df[tech_cols].ffill().bfill()
        if not tech_data.empty and tech_data.notna().sum().sum() > 0:
            create_single_heatmap(f"{dataset_name} - Technology Bands", tech_data,
                                  dataset_folder, "Technology bands", "EMF_")
        else:
            print(f"   No valid technology data for {dataset_name}")
    else:
        print(f"   Insufficient technology columns for {dataset_name} ({len(tech_cols)} found)")

    # 3. Operator aggregation heatmap.
    operator_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                     and any(op in col for op in ['Iliad', 'TIM', 'VF', 'W3'])]
    if len(operator_cols) >= 2:
        operator_data = df[operator_cols].ffill().bfill()
        if not operator_data.empty and operator_data.notna().sum().sum() > 0:
            create_single_heatmap(f"{dataset_name} - Operators", operator_data,
                                  dataset_folder, "Operators", "EMF_")
        else:
            print(f"   No valid operator data for {dataset_name}")
    else:
        print(f"   Insufficient operator columns for {dataset_name} ({len(operator_cols)} found)")

def create_dataset_trio_heatmap(df, dataset_name, dataset_folder):
    """Create a 1x3 heatmap (Frequency, Technology, Operator) with a shared legend."""
    dataset_folder.mkdir(parents=True, exist_ok=True)

    # Get data for each type
    freq_cols = get_actual_frequency_columns(df)
    tech_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_') 
                 and any(tech in col for tech in ['2G', '3G', '4G', '5G'])]
    operator_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_') 
                     and any(op in col for op in ['Iliad', 'TIM', 'VF', 'W3'])]
    
    # Check if we have data for all three
    if len(freq_cols) < 2 or len(tech_cols) < 2 or len(operator_cols) < 2:
        print(f"   ⚠️  Skipping trio heatmap for {dataset_name}: insufficient data")
        return
    
    # Create 1x3 subplot figure with space for common colorbar - larger figure for clarity
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    plot_data = [
        (freq_cols, "(a) Frequency Bands", None, axes[0], False),
        (tech_cols, "(b) Technology Bands", "EMF_", axes[1], True),
        (operator_cols, "(c) Operator Bands", "EMF_", axes[2], True)
    ]
    
    correlation_matrices = []
    
    for cols_to_use, subplot_title, trim_prefix, ax, is_trimmed in plot_data:
        subset = df[cols_to_use].ffill().bfill()
        if not subset.empty and subset.notna().sum().sum() > 0:
            corr = subset.corr()
            correlation_matrices.append(corr)
            
            # Prepare labels
            labels = list(corr.columns)
            if trim_prefix:
                labels = [str(c).replace(trim_prefix, "") for c in labels]
            
            # Plot without colorbar (will add shared one later)
            im = ax.imshow(corr, cmap='viridis', vmin=0, vmax=1)
            
            # Set ticks and labels with dynamic font sizing
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            
            # Adjust font size based on number of items - enlarge text for clarity
            if len(labels) > 15:  # High density (frequencies)
                label_fontsize = 11
                tick_fontsize = 10
                text_fontsize = 8
            elif len(labels) > 8:  # Medium density
                label_fontsize = 12
                tick_fontsize = 11
                text_fontsize = 9
            else:  # Low density (technology/operators)
                label_fontsize = 13
                tick_fontsize = 12
                text_fontsize = 12
            
            ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
            ax.set_yticklabels(labels, fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            
            # Add correlation values with BLACK text (enlarged)
            # 12pt for trimmed data (tech/operator), 6pt for frequencies
            if len(labels) <= 10:
                text_size = 12 if is_trimmed else 6
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                               ha="center", va="center", fontsize=text_size, 
                               color='black', fontweight='bold')
            
            # Add subplot label below
            ax.set_xlabel(subplot_title, fontsize=13, fontweight='bold', labelpad=10)
        else:
            ax.text(0.5, 0.5, f"No data\navailable", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            correlation_matrices.append(None)
            ax.set_xlabel(subplot_title, fontsize=13, fontweight='bold', labelpad=10)
    
    # Add common colorbar on the right - positioned closer to subplots
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Correlation', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Reduce spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.15, wspace=0.2)
    
    # Save the combined trio plot.
    name = "Combined (frequency, technology, operators)"
    fig.savefig(dataset_folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(dataset_folder / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"   Saved: {name}.png")
    plt.close(fig)

def create_all_datasets_comparison_heatmap(datasets_info, output_root):
    """Create a comparison heatmap showing all datasets in one view."""
    output_root.mkdir(parents=True, exist_ok=True)

    if len(datasets_info) < 2:
        print("   Need at least 2 datasets for comparison heatmap")
        return
    
    # Create 3 x len(datasets) subplot
    n_datasets = len(datasets_info)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(18, 6*n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('EMF Data Correlation Analysis: All Datasets Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    for row, (df, dataset_name, _folder) in enumerate(datasets_info):
        # Get data for each type
        freq_cols = get_actual_frequency_columns(df)
        tech_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_') 
                     and any(tech in col for tech in ['2G', '3G', '4G', '5G'])]
        operator_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_') 
                         and any(op in col for op in ['Iliad', 'TIM', 'VF', 'W3'])]
        
        # Plot types: Frequencies, Technology, Operators
        plot_data = [
            (freq_cols, "Frequency Bands", None),
            (tech_cols, "Technology Bands", "EMF_"),
            (operator_cols, "Operators", "EMF_")
        ]
        
        for col, (cols_to_use, comp_name, trim_prefix) in enumerate(plot_data):
            ax = axes[row, col]
            
            if len(cols_to_use) >= 2:
                subset = df[cols_to_use].ffill().bfill()
                if not subset.empty and subset.notna().sum().sum() > 0:
                    corr = subset.corr()
                    
                    im = ax.imshow(corr, cmap='viridis', vmin=0, vmax=1)
                    
                    # Prepare labels with ENLARGED fonts
                    labels = list(corr.columns)
                    if trim_prefix:
                        labels = [str(c).replace(trim_prefix, "") for c in labels]
                    
                    # Set ticks and labels with dynamic font sizing
                    ax.set_xticks(np.arange(len(labels)))
                    ax.set_yticks(np.arange(len(labels)))
                    
                    # Adjust font size based on number of items (smart crowding detection)
                    if len(labels) > 15:  # High density (frequencies)
                        label_fontsize = 15# 7
                        tick_fontsize = 15# 6
                        text_fontsize = 15#5
                    elif len(labels) > 8:  # Medium density
                        label_fontsize = 15#8
                        tick_fontsize = 15#7
                        text_fontsize = 15#6
                    else:  # Low density (technology/operators)
                        label_fontsize = 15#11
                        tick_fontsize = 15#11
                        text_fontsize = 15#9
                    
                    ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
                    ax.set_yticklabels(labels, fontsize=label_fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    
                    # Add colorbar with appropriate font size
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=tick_fontsize)
                    
                    # Add correlation values for smaller matrices
                    if len(labels) <= 10:
                        for i in range(corr.shape[0]):
                            for j in range(corr.shape[1]):
                                ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                                       ha="center", va="center", fontsize=text_fontsize, 
                                       color='white', fontweight='bold')
                    
                    ax.set_title(f"{dataset_name}\n{comp_name}", fontsize=13, fontweight='bold', pad=10)
                else:
                    ax.text(0.5, 0.5, f"No data\navailable", ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10)
                    ax.set_title(f"{dataset_name}\n{comp_name}", fontsize=10, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, f"Insufficient\ndata\n({len(cols_to_use)} cols)", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{dataset_name}\n{comp_name}", fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save the comparison grid at the Heatmaps root.
    name = "All datasets comparison"
    fig.savefig(output_root / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_root / f"{name}.pdf", dpi=300, bbox_inches="tight")
    print(f"   Saved: {name}.png")
    plt.close(fig)

def generate_all_heatmaps(datasets_info, output_root):
    """Generate all heatmaps for the given datasets.

    Args:
        datasets_info: list of (dataframe, dataset_name, folder_name) tuples
        output_root: Path of the Heatmaps root folder
    """
    print("Generating correlation heatmaps...")

    for df, dataset_name, folder in datasets_info:
        generate_dataset_heatmaps(df, dataset_name, output_root / folder)

    create_all_datasets_comparison_heatmap(datasets_info, output_root)

    print(f"All heatmaps saved in: {output_root}")