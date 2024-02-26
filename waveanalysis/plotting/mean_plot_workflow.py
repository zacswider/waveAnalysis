import numpy as np
from .mean_plot_creation import return_mean_ACF_figure

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