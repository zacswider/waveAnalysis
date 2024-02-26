import numpy as np
from .mean_plot_creation import return_mean_ACF_figure, return_mean_prop_peaks_figure

# TODO: create workflows for the mean plots

def plot_mean_ACFs_workflow(
    acfs: np.ndarray, 
    periods: np.ndarray, 
    num_channels: int,
    num_frames: int
) -> dict:
    # Dictionary to store generated figures
    acf_figs = {}
    
    # Generate plots for each channel
    for channel in range(num_channels):
        acf_figs[f'Ch{channel + 1} Mean ACF'] = return_mean_ACF_figure(
            signal=acfs[channel], 
            shifts_or_periods=periods[channel], 
            channel=f'Ch{channel + 1}',
            num_frames= num_frames)     
           
    return acf_figs


def plot_mean_prop_peaks_workflow(
    indv_peak_mins: np.ndarray, 
    indv_peak_maxs: np.ndarray, 
    indv_peak_amps: np.ndarray,
    indv_peak_widths: np.ndarray,
    num_channels: int
) -> dict:
    # Empty dictionary to fill with figures for each channel
    peak_figs = {}

    for channel in range(num_channels):
        peak_figs[f'Ch{channel + 1} Peak Props'] = return_mean_prop_peaks_figure(
            min_array=indv_peak_mins[channel], 
            max_array=indv_peak_maxs[channel], 
            amp_array=indv_peak_amps[channel], 
            width_array=indv_peak_widths[channel], 
            Ch_name=f'Ch{channel + 1}')

    return peak_figs