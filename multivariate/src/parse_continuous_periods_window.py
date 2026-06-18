#!/usr/bin/env python3
# -*- coding: utf‑8 -*-

"""
Parse Continuous Periods Window Analysis for fileCP_2024_UNIVERSITY_HOSPITAL
Creates zoom-in plots for different time windows (1, 3, 7, 14 days) from the middle of the longest continuous period.
Continuous period: 2023-07-26 16:49:02 to 2024-03-11 05:08:28 (228 days, 12 hours)
Window center date: 2023-11-17 (middle of the period)

Time windows analyzed:
- 1 day: 2023-11-17 00:00:00 to 2023-11-17 23:59:59
- 3 days: 2023-11-16 00:00:00 to 2023-11-18 23:59:59  
- 7 days: 2023-11-14 00:00:00 to 2023-11-20 23:59:59
- 14 days: 2023-11-10 00:00:00 to 2023-11-23 23:59:59
"""

import math
import re
import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set global font to Times New Roman with larger sizes
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20

# Define the target dataset and its continuous period
TARGET_DATASET = "fileCP_2024_UNIVERSITY_HOSPITAL.txt"
CONTINUOUS_PERIOD = {
    "start": dt.datetime(2023, 7, 26, 16, 49, 2),
    "end": dt.datetime(2024, 3, 11, 5, 8, 28),
    "duration": "228 days, 12 hours"
}

# Define window center date (middle of continuous period)
WINDOW_CENTER = dt.datetime(2023, 11, 17, 12, 0, 0)  # November 17, 2023, noon

# Define time windows (in days)
WINDOW_SIZES = {
    "1day": 1,
    "3days": 3, 
    "7days": 7,
    "14days": 14
}

def parse_raw(path: Path) -> pd.DataFrame:
    """Return DataFrame indexed by Timestamp, freq columns (float), excluding all-zero rows."""
    print(f"📄 Parsing: {path.name}")
    
    txt = path.read_text(encoding="utf‑8", errors="ignore")
    rows = []
    all_zero_count = 0

    for block in txt.split("Inizio;")[1:]:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 4:
            continue

        try:
            t0 = dt.datetime.strptime(lines[0], "%d/%m/%Y %H:%M:%S")
        except ValueError:
            continue

        freqs = [
            float(m.group())
            for tok in lines[1].split(";")
            for m in [re.search(r"\d+(?:\.\d+)?", tok)]
            if m
        ]
        n = len(freqs)

        for line in lines[3:]:
            tok = [t for t in line.split(";") if t]
            if len(tok) < 2 * n:
                continue

            try:
                row_ts = dt.datetime.strptime(tok[0], "%d/%m/%Y %H:%M:%S")
            except ValueError:
                continue

            row = {"Timestamp": row_ts}
            valid_values = []
            all_zero = True
            
            for i, f in enumerate(freqs):
                try:
                    v = float(tok[2 * i + 1])
                    if v == 0:
                        v = math.nan
                    else:
                        valid_values.append(v)
                        all_zero = False
                except ValueError:
                    v = math.nan
                row[f] = v
            
            # Only include row if there's at least one non-zero measurement
            if not all_zero and valid_values:
                rows.append(row)
            else:
                all_zero_count += 1

    df = (
        pd.DataFrame(rows)
          .set_index("Timestamp")
          .sort_index()
    )
    
    print(f"   ✅ Parsed {len(df):,} rows with {len(df.columns)} frequencies")
    print(f"   🔄 Filtered out {all_zero_count:,} all-zero rows")
    return df

def filter_to_continuous_period(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to the continuous period."""
    start_time = CONTINUOUS_PERIOD["start"]
    end_time = CONTINUOUS_PERIOD["end"]
    
    print(f"   🔄 Filtering to continuous period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📅 Period duration: {CONTINUOUS_PERIOD['duration']}")
    
    # Filter data to the continuous period
    filtered_df = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    
    original_rows = len(df)
    filtered_rows = len(filtered_df)
    
    print(f"   ✅ Filtered: {original_rows:,} → {filtered_rows:,} rows ({filtered_rows/original_rows*100:.1f}% retained)")
    
    return filtered_df

def load_antenna_factors():
    """Load antenna factor calibration data."""
    AF_FILE = Path("9018-18835.xlsx")
    if not AF_FILE.exists():
        print(f"   ⚠️  Antenna factor file not found: {AF_FILE}")
        return None, None
    
    try:
        af_df = pd.read_excel(AF_FILE, usecols=["Freq (MHz)", "AF(dBm-1)"]).dropna()
        af_freq = af_df["Freq (MHz)"].to_numpy()
        af_val = af_df["AF(dBm-1)"].to_numpy()
        print(f"   📊 Loaded antenna factors: {len(af_freq)} frequency points")
        return af_freq, af_val
    except Exception as e:
        print(f"   ⚠️  Error loading antenna factors: {str(e)}")
        return None, None

def get_antenna_factor(freq_mhz: float, af_freq, af_val) -> float:
    """Get nearest-neighbor antenna factor in dB m⁻¹."""
    if af_freq is None or af_val is None:
        return 20.0  # Default fallback value
    idx = np.abs(af_freq - freq_mhz).argmin()
    return float(af_val[idx])

def dBm_to_E_field(dBm: float, freq_mhz: float, af_freq, af_val) -> float:
    """Convert dBm to E-field (V/m) using antenna factors."""
    if math.isnan(dBm):
        return math.nan
    
    # Constants
    C = 299_792_458.0  # speed of light
    Z0 = 120 * math.pi  # free-space impedance
    
    # Received power
    P_r = 10 ** ((dBm - 30) / 10)  # W
    
    # Wavelength & antenna gain
    λ = C / (freq_mhz * 1e6)  # m
    α_db = get_antenna_factor(freq_mhz, af_freq, af_val)  # dB m⁻¹
    G_f = (9.73 / (λ * 10 ** (α_db / 20)))**2  # linear gain
    
    # Effective aperture
    A_e = G_f * λ ** 2 / (4 * math.pi)  # m²
    
    # Return actual E-field (V/m)
    return math.sqrt(P_r * Z0 / A_e)

def convert_to_emf(rss_df: pd.DataFrame, af_freq, af_val) -> pd.DataFrame:
    """Convert RSS DataFrame to EMF DataFrame."""
    print(f"   🔄 Converting dBm to E-field...")
    
    emf_df = rss_df.copy()
    freq_cols = [col for col in rss_df.columns if isinstance(col, (int, float))]
    
    for freq in freq_cols:
        emf_df[freq] = rss_df[freq].apply(
            lambda dBm: dBm_to_E_field(dBm, freq, af_freq, af_val)
        )
    
    print(f"   ✅ EMF conversion complete for {len(freq_cols)} frequencies")
    return emf_df

def add_aggregated_columns(emf_df: pd.DataFrame) -> pd.DataFrame:
    """Add technology, operator, and total EMF aggregation columns."""
    print(f"   📊 Adding aggregated EMF columns...")
    
    # Technology bands (based on fileCP_2024_UNIVERSITY_HOSPITAL frequencies)
    TECH_BANDS = {
        "2G": [935.0, 945.0, 955.0],
        "3G": [2120.0, 2137.5, 2150.0, 2162.5],
        "4G": [
            763.0, 773.0, 783.0, 796.0, 806.0, 816.0,
            1835.0, 1850.0, 1870.0,
            2585.0, 2635.0, 2647.5, 2662.5, 2680.0
        ],
        "5G": [
            3449.0, 3469.0, 3489.0, 3547.5, 3589.0, 3630.0, 3680.0, 3760.0
        ]
    }
    
    # Operator bands (based on fileCP_2024_UNIVERSITY_HOSPITAL frequencies)
    OPERATOR_BANDS = {
        "Iliad": [763.0, 1835.0, 2150.0, 2635.0, 3630.0],
        "TIM": [773.0, 806.0, 935.0, 2137.5, 2662.5, 3760.0, 3469.0],
        "VF": [783.0, 816.0, 945.0, 1870.0, 2162.5, 2647.5, 3680.0],
        "W3": [796.0, 955.0, 1850.0, 2120.0, 2585.0, 2680.0, 3610.0]
    }
    
    # Add technology aggregations
    for tech, freqs in TECH_BANDS.items():
        present_freqs = [f for f in freqs if f in emf_df.columns]
        if present_freqs:
            emf_val = (emf_df[present_freqs] ** 2).sum(axis=1).pow(0.5)
            emf_val = emf_val.where(emf_val != 0, np.nan)
            emf_df[f"EMF_{tech}"] = emf_val
        else:
            print(f"   ⚠️  No frequencies found for {tech}")
    
    # Add operator aggregations
    for op, freqs in OPERATOR_BANDS.items():
        present_freqs = [f for f in freqs if f in emf_df.columns]
        if present_freqs:
            emf_val = (emf_df[present_freqs] ** 2).sum(axis=1).pow(0.5)
            emf_val = emf_val.where(emf_val != 0, np.nan)
            emf_df[f"EMF_{op}"] = emf_val
        else:
            print(f"   ⚠️  No frequencies found for {op}")
    
    # Add total EMF across all frequencies
    freq_cols = [c for c in emf_df.columns if isinstance(c, (int, float))]
    if freq_cols:
        emf_val = (emf_df[freq_cols] ** 2).sum(axis=1).pow(0.5)
        emf_val = emf_val.where(emf_val != 0, np.nan)
        emf_df["EMF_Total"] = emf_val
    
    # Count aggregated columns
    tech_cols = [col for col in emf_df.columns if isinstance(col, str) and col.startswith('EMF_') and any(tech in col for tech in TECH_BANDS.keys())]
    op_cols = [col for col in emf_df.columns if isinstance(col, str) and col.startswith('EMF_') and any(op in col for op in OPERATOR_BANDS.keys())]
    total_cols = [col for col in emf_df.columns if col == 'EMF_Total']
    
    print(f"   ✅ Added {len(tech_cols)} technology + {len(op_cols)} operator + {len(total_cols)} total EMF columns")
    return emf_df

def get_window_timerange(window_name: str, center_date: dt.datetime):
    """Calculate start and end times for a given window size centered on center_date."""
    days = WINDOW_SIZES[window_name]
    half_window = dt.timedelta(days=days/2)
    
    # For day-based windows, align to midnight boundaries
    start_time = (center_date - half_window).replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = (center_date + half_window).replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return start_time, end_time

def filter_to_window(df: pd.DataFrame, window_name: str) -> pd.DataFrame:
    """Filter DataFrame to specific time window."""
    start_time, end_time = get_window_timerange(window_name, WINDOW_CENTER)
    
    print(f"   🪟 Filtering to {window_name} window: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Filter data to the window
    windowed_df = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    
    original_rows = len(df)
    windowed_rows = len(windowed_df)
    
    print(f"   ✅ Window filtered: {original_rows:,} → {windowed_rows:,} rows")
    
    return windowed_df

def remove_outliers_for_plotting(df, cols, n_std=3):
    """Remove rows where any selected column is outside n_std standard deviations."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col in cols:
        if col in df:
            mean = df[col].mean()
            std = df[col].std()
            mask &= (df[col] >= mean - n_std * std) & (df[col] <= mean + n_std * std)
    return df[mask]

def plot_window_analysis(emf_df: pd.DataFrame, output_folder: Path):
    """Generate window analysis plots for different time windows."""
    print(f"📊 Generating window analysis plots...")
    
    # Create plots subfolder
    plots_folder = output_folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    
    # Define column groups
    TECH_COLS = ["EMF_2G", "EMF_3G", "EMF_4G", "EMF_5G"]
    OP_COLS = ["EMF_W3", "EMF_Iliad", "EMF_TIM", "EMF_VF"]
    
    # Check which columns exist
    available_tech_cols = [col for col in TECH_COLS if col in emf_df.columns]
    available_op_cols = [col for col in OP_COLS if col in emf_df.columns]
    
    if not available_tech_cols and not available_op_cols:
        print(f"   ⚠️  No EMF aggregation columns found for plotting")
        return
    
    print(f"   📊 Available technology columns: {available_tech_cols}")
    print(f"   📊 Available operator columns: {available_op_cols}")
    
    # Process each window size
    for window_name in WINDOW_SIZES.keys():
        print(f"\n🪟 Processing {window_name} window...")
        
        # Filter data to window
        windowed_df = filter_to_window(emf_df, window_name)
        
        if windowed_df.empty:
            print(f"   ⚠️  No data in {window_name} window")
            continue
            
        # Reset index for plotting
        df_plot = windowed_df.reset_index()
        
        # Get window timerange for title
        start_time, end_time = get_window_timerange(window_name, WINDOW_CENTER)
        
        # Plot technology bands
        if available_tech_cols:
            plt.figure(figsize=(16, 8))
            df_clean = remove_outliers_for_plotting(df_plot, available_tech_cols)
            
            for col in available_tech_cols:
                legend_label = col.replace("EMF_", "")
                plt.plot(df_clean["Timestamp"], df_clean[col], lw=0.8, alpha=0.8, label=legend_label)
            
            plt.xlabel("Timestamp", fontsize=20, fontweight='bold')
            plt.ylabel("EMF (V/m)", fontsize=20, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.00),
                ncol=len(available_tech_cols),
                frameon=False,
                fontsize=16,
                title_fontsize=18,
                prop={'weight': 'bold'}
            )
            # plt.title(f"EMF Technology Bands - {window_name.upper()} Window\n{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}", 
            #          fontsize=18, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save technology plot
            tech_png = plots_folder / f"emf_technology_2024_UNIVERSITY_HOSPITAL_{window_name}_window.png"
            tech_pdf = plots_folder / f"emf_technology_2024_UNIVERSITY_HOSPITAL_{window_name}_window.pdf"
            plt.savefig(tech_png, dpi=300, bbox_inches='tight')
            plt.savefig(tech_pdf, dpi=300, format='pdf', bbox_inches='tight')
            print(f"   ✅ Saved technology plot: {tech_png.name}")
            plt.close()
        
        # Plot operator bands  
        if available_op_cols:
            plt.figure(figsize=(16, 8))
            df_clean = remove_outliers_for_plotting(df_plot, available_op_cols)
            
            for col in available_op_cols:
                legend_label = col.replace("EMF_", "")
                plt.plot(df_clean["Timestamp"], df_clean[col], lw=0.8, alpha=0.8, label=legend_label)
            
            plt.xlabel("Timestamp", fontsize=20, fontweight='bold')
            plt.ylabel("EMF (V/m)", fontsize=20, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.00),
                ncol=len(available_op_cols),
                frameon=False,
                fontsize=16,
                title_fontsize=18,
                prop={'weight': 'bold'}
            )
            # plt.title(f"EMF Operator Bands - {window_name.upper()} Window\n{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}", 
            #          fontsize=18, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save operator plot
            op_png = plots_folder / f"emf_operator_2024_UNIVERSITY_HOSPITAL_{window_name}_window.png"
            op_pdf = plots_folder / f"emf_operator_2024_UNIVERSITY_HOSPITAL_{window_name}_window.pdf"
            plt.savefig(op_png, dpi=300, bbox_inches='tight')
            plt.savefig(op_pdf, dpi=300, format='pdf', bbox_inches='tight')
            print(f"   ✅ Saved operator plot: {op_png.name}")
            plt.close()

def plot_all_windows_summary(emf_df: pd.DataFrame, output_folder: Path):
    """Generate a summary plot showing all window sizes together."""
    print(f"📊 Generating all windows summary plot...")
    
    # Create plots subfolder
    plots_folder = output_folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    
    # Define column groups
    TECH_COLS = ["EMF_2G", "EMF_3G", "EMF_4G", "EMF_5G"]
    OP_COLS = ["EMF_W3", "EMF_Iliad", "EMF_TIM", "EMF_VF"]
    
    # Check which columns exist
    available_tech_cols = [col for col in TECH_COLS if col in emf_df.columns]
    available_op_cols = [col for col in OP_COLS if col in emf_df.columns]
    
    if not available_tech_cols and not available_op_cols:
        print(f"   ⚠️  No EMF aggregation columns found for summary plotting")
        return
    
    # Create figure with 4 rows (windows) and 2 columns (tech/operator)
    fig, axes = plt.subplots(4, 2, figsize=(24, 20))
    
    fig.suptitle('EMF Window Analysis - 2024 UNIVERSITY HOSPITAL\nMultiple Time Windows from Continuous Period', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    window_names = list(WINDOW_SIZES.keys())
    
    for row, window_name in enumerate(window_names):
        # Filter data to window
        windowed_df = filter_to_window(emf_df, window_name)
        
        if windowed_df.empty:
            # Handle empty window
            for col in [0, 1]:
                axes[row, col].text(0.5, 0.5, f"No data in {window_name} window", 
                                  ha='center', va='center', transform=axes[row, col].transAxes, fontsize=14)
                axes[row, col].set_title(f"{window_name.upper()} Window", fontsize=16, fontweight='bold')
            continue
            
        # Reset index for plotting
        df_plot = windowed_df.reset_index()
        start_time, end_time = get_window_timerange(window_name, WINDOW_CENTER)
        
        # Plot technology bands (left column)
        ax_tech = axes[row, 0]
        if available_tech_cols:
            df_clean = remove_outliers_for_plotting(df_plot, available_tech_cols)
            
            for col in available_tech_cols:
                legend_label = col.replace("EMF_", "")
                ax_tech.plot(df_clean["Timestamp"], df_clean[col], lw=0.6, alpha=0.8, label=legend_label)
            
            ax_tech.set_xlabel("Timestamp", fontsize=14, fontweight='bold')
            ax_tech.set_ylabel("EMF (V/m)", fontsize=14, fontweight='bold')
            ax_tech.tick_params(axis='both', which='major', labelsize=12)
            ax_tech.legend(loc='upper right', fontsize=10, frameon=True, prop={'weight': 'bold'})
            ax_tech.set_title(f"Technology - {window_name.upper()} ({start_time.strftime('%m-%d')} to {end_time.strftime('%m-%d')})", fontsize=14, fontweight='bold')
            ax_tech.grid(True, alpha=0.3)
        else:
            ax_tech.text(0.5, 0.5, f"No Technology\nData Available", 
                        ha='center', va='center', transform=ax_tech.transAxes, fontsize=12)
            ax_tech.set_title(f"Technology - {window_name.upper()}", fontsize=14, fontweight='bold')
        
        # Plot operator bands (right column)
        ax_op = axes[row, 1]
        if available_op_cols:
            df_clean = remove_outliers_for_plotting(df_plot, available_op_cols)
            
            for col in available_op_cols:
                legend_label = col.replace("EMF_", "")
                ax_op.plot(df_clean["Timestamp"], df_clean[col], lw=0.6, alpha=0.8, label=legend_label)
            
            ax_op.set_xlabel("Timestamp", fontsize=14, fontweight='bold')
            ax_op.set_ylabel("EMF (V/m)", fontsize=14, fontweight='bold')
            ax_op.tick_params(axis='both', which='major', labelsize=12)
            ax_op.legend(loc='upper right', fontsize=10, frameon=True, prop={'weight': 'bold'})
            ax_op.set_title(f"Operator - {window_name.upper()} ({start_time.strftime('%m-%d')} to {end_time.strftime('%m-%d')})", fontsize=14, fontweight='bold')
            ax_op.grid(True, alpha=0.3)
        else:
            ax_op.text(0.5, 0.5, f"No Operator\nData Available", 
                      ha='center', va='center', transform=ax_op.transAxes, fontsize=12)
            ax_op.set_title(f"Operator - {window_name.upper()}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the all windows summary plot
    summary_png = plots_folder / "emf_2024_UNIVERSITY_HOSPITAL_all_windows_summary.png"
    summary_pdf = plots_folder / "emf_2024_UNIVERSITY_HOSPITAL_all_windows_summary.pdf"
    
    fig.savefig(summary_png, dpi=300, bbox_inches='tight')
    fig.savefig(summary_pdf, dpi=300, format='pdf', bbox_inches='tight')
    print(f"   ✅ Saved all windows summary plot: {summary_png.name}")
    print(f"   ✅ Saved all windows summary plot: {summary_pdf.name}")
    plt.close(fig)

def main():
    print("🪟 FileCP Window Analysis - 2024 UNIVERSITY HOSPITAL")
    print("="*70)
    print(f"📁 Target dataset: {TARGET_DATASET}")
    print(f"📅 Continuous period: {CONTINUOUS_PERIOD['start'].strftime('%Y-%m-%d %H:%M:%S')} to {CONTINUOUS_PERIOD['end'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Window center: {WINDOW_CENTER.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display window information
    print(f"\n🪟 Window Analysis:")
    for window_name, days in WINDOW_SIZES.items():
        start_time, end_time = get_window_timerange(window_name, WINDOW_CENTER)
        print(f"   • {window_name}: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({days} days)")
    
    # Create result folder with timestamp subfolder
    result_folder = Path("result")
    result_folder.mkdir(exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_folder = result_folder / f"window_analysis_{timestamp}"
    timestamped_folder.mkdir(exist_ok=True)
    
    print(f"\n📁 Output folder: {timestamped_folder}")
    
    # Check if dataset file exists
    file_path = Path(TARGET_DATASET)
    if not file_path.exists():
        print(f"❌ File not found: {TARGET_DATASET}")
        return
    
    print("\n" + "="*70)
    
    try:
        # Parse the dataset
        df = parse_raw(file_path)
        
        if df.empty:
            print(f"❌ No data extracted from: {TARGET_DATASET}")
            return
        
        # Filter to continuous period
        df = filter_to_continuous_period(df)
        
        if df.empty:
            print(f"❌ No data in continuous period")
            return
        
        # Load antenna factors
        print(f"\n📊 Loading antenna factor calibration...")
        af_freq, af_val = load_antenna_factors()
        
        # Convert RSS to EMF
        emf_df = convert_to_emf(df, af_freq, af_val)
        
        # Add aggregated EMF columns (technology, operator, total)
        emf_df = add_aggregated_columns(emf_df)
        
        # Count different column types
        freq_cols = [col for col in emf_df.columns if isinstance(col, (int, float))]
        emf_agg_cols = [col for col in emf_df.columns if isinstance(col, str) and col.startswith('EMF_')]
        
        print(f"   📊 Dataset ready: {len(freq_cols)} frequencies + {len(emf_agg_cols)} EMF aggregations")
        print(f"   📊 Total data points in continuous period: {len(emf_df):,}")
        
        # Generate window analysis plots
        plot_window_analysis(emf_df, timestamped_folder)
        
        # Generate all windows summary plot
        plot_all_windows_summary(emf_df, timestamped_folder)
        
        # Generate summary report
        summary_file = timestamped_folder / "window_analysis_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("FileCP Window Analysis Summary - 2024 UNIVERSITY HOSPITAL\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output folder: {timestamped_folder}\n\n")
            
            f.write(f"TARGET DATASET: {TARGET_DATASET}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Continuous period: {CONTINUOUS_PERIOD['start'].strftime('%Y-%m-%d %H:%M:%S')} to {CONTINUOUS_PERIOD['end'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {CONTINUOUS_PERIOD['duration']}\n")
            f.write(f"Window center: {WINDOW_CENTER.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total data points: {len(emf_df):,}\n\n")
            
            f.write("WINDOW ANALYSIS:\n")
            f.write("-" * 60 + "\n")
            for window_name, days in WINDOW_SIZES.items():
                start_time, end_time = get_window_timerange(window_name, WINDOW_CENTER)
                windowed_df = filter_to_window(emf_df, window_name)
                f.write(f"{window_name}: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({days} days)\n")
                f.write(f"  Data points: {len(windowed_df):,}\n")
                f.write(f"  Files generated:\n")
                f.write(f"    - emf_technology_2024_UNIVERSITY_HOSPITAL_{window_name}_window.png/.pdf\n")
                f.write(f"    - emf_operator_2024_UNIVERSITY_HOSPITAL_{window_name}_window.png/.pdf\n\n")
            
            f.write("SUMMARY VISUALIZATIONS:\n")
            f.write("-" * 60 + "\n")
            f.write("• emf_2024_UNIVERSITY_HOSPITAL_all_windows_summary.png/.pdf (4 windows × 2 plot types)\n")
        
        # Final summary
        print("\n" + "="*70)
        print("📋 WINDOW ANALYSIS SUMMARY")
        print("="*70)
        print(f"✅ Successfully processed: {TARGET_DATASET}")
        print(f"📁 Output folder: {timestamped_folder}")
        print(f"📊 Plots folder: {timestamped_folder}/plots")
        print(f"📄 Summary report: {summary_file.name}")
        
        print(f"\n📊 Generated window plots:")
        for window_name in WINDOW_SIZES.keys():
            print(f"   • {window_name} window:")
            print(f"     - emf_technology_2024_UNIVERSITY_HOSPITAL_{window_name}_window.png/.pdf")
            print(f"     - emf_operator_2024_UNIVERSITY_HOSPITAL_{window_name}_window.png/.pdf")
        
        print(f"\n📊 Summary visualization:")
        print(f"   • emf_2024_UNIVERSITY_HOSPITAL_all_windows_summary.png/.pdf (4 windows × 2 plot types)")
        
        print(f"\n🎉 Window analysis complete! Files saved in: {timestamped_folder.absolute()}")
        
    except Exception as e:
        print(f"❌ Error processing {TARGET_DATASET}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()