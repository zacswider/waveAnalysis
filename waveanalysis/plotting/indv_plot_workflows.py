from tqdm import tqdm
import numpy as np
from .indv_figure_creation import return_indv_peak_prop_figure, return_indv_acf_figure, return_indv_ccf_figure
from waveanalysis.signal_processing import normalize_signal
from itertools import zip_longest


def plot_indv_peak_props_workflow(
    num_channels:int,
    total_bins:int,
    bin_values:np.ndarray,
    analysis_type:str,
    ind_peak_props:dict
) -> dict:

    # Dictionary to store generated figures
    indv_peak_figs = {}

    # Generate plots for each channel

    its = num_channels*total_bins
    with tqdm(total=its, miniters=its/100) as pbar:
        pbar.set_description('ind peaks')
        for channel in range(num_channels):
            for bin in range(total_bins):
                pbar.update(1)
                to_plot = bin_values[:,channel, bin] if analysis_type == "standard" else bin_values[channel,bin, :]
                # Generate and store the figure for the current channel and bin
                indv_peak_figs[f'Ch{channel + 1} Bin {bin + 1} Peak Props'] = return_indv_peak_prop_figure(
                    bin_signal=to_plot,
                    prop_dict=ind_peak_props[f'Ch {channel} Bin {bin}'],
                    Ch_name=f'Ch{channel + 1} Bin {bin + 1}'
                    )

    return indv_peak_figs

def plot_indv_acfs_workflow(
    num_channels:int,
    total_bins:int,
    bin_values:np.ndarray,
    analysis_type:str,
    acfs:np.ndarray,
    periods:np.ndarray,
    num_frames:int
  )  -> dict:
        
        # Empty dictionary to store generated figures
        indv_acf_plots = {}

        # Iterate through channels and bins to plot individual autocorrelation curves
        its = num_channels*total_bins
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('ind acfs')
            for channel in range(num_channels):
                for bin in range(total_bins):
                    pbar.update(1) 
                    to_plot = bin_values[:,channel, bin] if analysis_type == "standard" else bin_values[channel,bin, :]
                    # Generate and store the figure for the current channel and bin
                    indv_acf_plots[f'Ch{channel + 1} Bin {bin + 1} ACF'] = return_indv_acf_figure(
                        raw_signal=to_plot, 
                        acf_curve=acfs[channel, bin], 
                        Ch_name=f'Ch{channel + 1}', 
                        period=periods[channel, bin],
                        num_frames= num_frames
                        )
                    
        return indv_acf_plots

def plot_indv_ccfs_workflow(
    total_bins:int,
    bin_values:np.ndarray,
    analysis_type:str,
    channel_combos:np.ndarray,
    indv_shifts:np.ndarray,
    indv_ccfs:np.ndarray,
    num_frames:int
) -> dict:
     # Empty dictionary to store generated figures
    indv_ccf_plots = {}

    # Iterate through channel combinations and bins to plot individual cross-correlation curves
    its = len(channel_combos)*total_bins
    with tqdm(total=its, miniters=its/100) as pbar:
        pbar.set_description('ind ccfs')
        for combo_number, combo in enumerate(channel_combos):
            for bin in range(total_bins):
                pbar.update(1)
                to_plot1 = bin_values[:, combo[0], bin] if analysis_type == "standard" else bin_values[combo[0], bin, :]
                to_plot2 = bin_values[:, combo[1], bin] if analysis_type == "standard" else bin_values[combo[1], bin, :]
                # Generate and store the figure for the current channel combination and bin
                indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = return_indv_ccf_figure(
                    ch1 = normalize_signal(to_plot1),
                    ch2 = normalize_signal(to_plot2),
                    ccf_curve = indv_ccfs[combo_number, bin],
                    ch1_name = f'Ch{combo[0] + 1}',
                    ch2_name = f'Ch{combo[1] + 1}',
                    shift = indv_shifts[combo_number, bin],
                    num_frames = num_frames)
    
    return indv_ccf_plots


# TODO: move this save folder when created

def save_indv_ccfs_workflow(
    indv_ccfs:np.ndarray,
    channel_combos:np.ndarray,
    bin_values:np.ndarray,
    analysis_type:str,
    total_bins:int
) -> dict:
    
    indv_ccf_values = {}

    for combo_number, combo in enumerate(channel_combos):
        for bin in range(total_bins):      
            # Save the individual bin values
            to_plot1 = bin_values[:, combo[0], bin] if analysis_type == "standard" else bin_values[combo[0], bin, :]
            to_plot2 = bin_values[:, combo[1], bin] if analysis_type == "standard" else bin_values[combo[1], bin, :]
            ccf_curve = indv_ccfs[combo_number, bin]
            measurements = list(zip_longest(range(1, len(ccf_curve) + 1),  normalize_signal(to_plot1), normalize_signal(to_plot2), ccf_curve, fillvalue=None))

            indv_ccf_values[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = measurements
            
    
    return indv_ccf_values
