import numpy as np
import matplotlib.pyplot as plt

# TODO: create functions for the mean plots

def return_mean_ACF_figure(
    signal: np.ndarray, 
    shifts_or_periods: np.ndarray, 
    channel: str,
    num_frames: int
) -> plt.Figure:

    # Plot mean autocorrelation curve with shaded area representing standard deviation
    signal_mean = np.nanmean(signal, axis = 0)
    signal_std = np.nanstd(signal, axis = 0)
    x_axis = np.arange(-num_frames + 1, num_frames)

    # Create the figure with subplots
    fig, ax = plt.subplot_mosaic(mosaic = '''
                                            AA
                                            BC
                                            ''')
    
    # Plot mean autocorrelation curve with shaded area representing standard deviation
    ax['A'].plot(x_axis, signal_mean, color='blue')
    ax['A'].fill_between(x_axis, 
                            signal_mean - signal_std, 
                            signal_mean + signal_std, 
                            color='blue', 
                            alpha=0.2)
    ax['A'].set_title(f'{channel} Mean Autocorrelation Curve ± Standard Deviation') 

    # Plot histogram of period values
    ax['B'].hist(shifts_or_periods)
    shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
    ax['B'].set_xlabel(f'Histogram of period values (frames)')
    ax['B'].set_ylabel('Occurances')

    # Plot boxplot of period values
    ax['C'].boxplot(shifts_or_periods)
    ax['C'].set_xlabel(f'Boxplot of period values')
    ax['C'].set_ylabel(f'Measured period (frames)')

    fig.subplots_adjust(hspace=0.25, wspace=0.5)  
    plt.close(fig)

    return fig