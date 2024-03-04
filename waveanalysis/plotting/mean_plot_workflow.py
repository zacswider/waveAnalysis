import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import zip_longest
from .mean_plot_creation import return_mean_ACF_figure, return_mean_prop_peaks_figure, return_mean_CCF_figure

# TODO: create workflows for the mean plots

def plot_mean_ACFs_workflow(
    acfs: np.ndarray, 
    periods: np.ndarray, 
    num_channels: int,
    num_frames: int
) -> dict:
    # Dictionary to store generated figures
    mean_acf_figs = {}
    
    # Generate plots for each channel
    for channel in range(num_channels):
        mean_acf_figs[f'Ch{channel + 1} Mean ACF'] = return_mean_ACF_figure(
            signal=acfs[channel], 
            shifts_or_periods=periods[channel], 
            channel=f'Ch{channel + 1}',
            num_frames= num_frames)     

    return mean_acf_figs

def plot_mean_prop_peaks_workflow(
    indv_peak_mins: np.ndarray, 
    indv_peak_maxs: np.ndarray, 
    indv_peak_amps: np.ndarray,
    indv_peak_widths: np.ndarray,
    num_channels: int
) -> dict:
    # Empty dictionary to fill with figures for each channel
    mean_peak_figs = {}

    for channel in range(num_channels):
        mean_peak_figs[f'Ch{channel + 1} Peak Props'] = return_mean_prop_peaks_figure(
            min_array=indv_peak_mins[channel], 
            max_array=indv_peak_maxs[channel], 
            amp_array=indv_peak_amps[channel], 
            width_array=indv_peak_widths[channel], 
            Ch_name=f'Ch{channel + 1}')

    return mean_peak_figs

def plot_mean_CCFs_workflow(
    signal: np.ndarray,
    shifts: np.ndarray,
    channel_combos: list,
    num_frames: int
) -> dict:
    # Empty dictionary to fill with figures for each channel
    mean_ccf_figs = {}

    # Iterate over each channel combination
    for combo_number, combo in enumerate(channel_combos):
        # Generate figure for mean CCF
        mean_ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_mean_CCF_figure(
            signal=signal[combo_number], 
            shifts=shifts[combo_number], 
            channel_combo=f'Ch{combo[0] + 1}-Ch{combo[1] + 1}',
            num_frames= num_frames)
    
    return mean_ccf_figs


# TODO: move this save folder when created
def save_mean_CCF_values_workflow(
    channel_combos: list, 
    indv_ccfs: np.ndarray, 
) -> dict:

    mean_ccf_values = {}

    for combo_number, combo in enumerate(channel_combos):
        arr_mean = np.nanmean(indv_ccfs[combo_number], axis = 0)
        arr_std = np.nanstd(indv_ccfs[combo_number], axis = 0)
        # Combine mean and standard deviation into a list of tuples
        mean_ccf_values[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF values'] = list(zip_longest(range(1, len(arr_mean) + 1), arr_mean, arr_std, fillvalue=None))

    return mean_ccf_values

def generate_group_comparison(
    summary_df: pd.DataFrame,
    log_params: dict
):
    print('Generating group comparisons...')
    
    mean_parameter_figs = {}
    parameters_to_compare = [column for column in summary_df.columns if 'Mean' in column]

    # generate and save figures for each parameter
    for param in parameters_to_compare:
        try:
            ax = sns.boxplot(x='Group Name', 
                            y=param, 
                            data=summary_df, 
                            palette = "Set2", 
                            showfliers = False)
            ax = sns.swarmplot(x='Group Name', 
                            y=param, 
                            data=summary_df, 
                            color=".25")	
            ax.set_xticklabels(ax.get_xticklabels(), 
                            rotation=45)
            fig = ax.get_figure()
            
            mean_parameter_figs[param] = fig
            plt.close(fig)

        except ValueError:
            log_params['Plotting errors'].append(f'No data to compare for {param}')

    return mean_parameter_figs