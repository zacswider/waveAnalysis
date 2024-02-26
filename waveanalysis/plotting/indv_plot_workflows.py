from tqdm import tqdm
import numpy as np
from .indv_figure_creation import return_indv_peak_prop_figure, return_indv_acf_figure

def plot_indv_peak_props_workflow(
    num_channels:int,
    total_bins:int,
    bin_values:np.ndarray,
    analysis_type:str,
    ind_peak_props:dict
):
    """
    This method generates and plots individual peak properties for each channel and bin.

    Returns:
        - dict: Dictionary containing generated figures of individual peak property plots.
    """
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
 ):
        """
        This method generates and plots individual autocorrelation functions (ACFs) for each channel and bin.

        Returns:
            - dict: Dictionary containing generated figures of individual ACF plots.
        """
        
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
