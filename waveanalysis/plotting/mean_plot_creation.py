import numpy as np
import matplotlib.pyplot as plt

# TODO: create functions for the mean plots

def return_mean_ACF_figure(
    signal: np.ndarray, 
    periods: np.ndarray, 
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
    ax['B'].hist(periods)
    periods = [val for val in periods if not np.isnan(val)]
    ax['B'].set_xlabel(f'Histogram of period values (frames)')
    ax['B'].set_ylabel('Occurances')

    # Plot boxplot of period values
    ax['C'].boxplot(periods)
    ax['C'].set_xlabel(f'Boxplot of period values')
    ax['C'].set_ylabel(f'Measured period (frames)')

    fig.subplots_adjust(hspace=0.25, wspace=0.5)  
    plt.close(fig)

    return fig

def return_mean_prop_peaks_figure(
    min_array: np.ndarray, 
    max_array: np.ndarray, 
    amp_array: np.ndarray, 
    width_array: np.ndarray,
    Ch_name: str
) -> plt.Figure:
    # Create subplots for histograms and boxplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Filter out NaN values from arrays
    min_array = [val for val in min_array if not np.isnan(val)]
    max_array = [val for val in max_array if not np.isnan(val)]
    amp_array = [val for val in amp_array if not np.isnan(val)]
    width_array = [val for val in width_array if not np.isnan(val)]

    # Define plot parameters for histograms and boxplots
    plot_params = { 'amp' : (amp_array, 'tab:blue'),
                    'min' : (min_array, 'tab:purple'),
                    'max' : (max_array, 'tab:orange')}
    
    # Plot histograms for peak properties
    for labels, (arr, arr_color) in plot_params.items():
        ax1.hist(arr, color = arr_color, label = labels, alpha = 0.75)

    # Plot boxplots for peak properties
    boxes = ax2.boxplot([val[0] for val in plot_params.values()], patch_artist = True)
    ax2.set_xticklabels(plot_params.keys())
    for box, box_color in zip(boxes['boxes'], [val[1] for val in plot_params.values()]):
        box.set_color(box_color)

    # Set labels and legends for histograms and boxplots
    ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
    ax1.set_xlabel(f'{Ch_name} histogram of peak values')
    ax1.set_ylabel('Occurances')
    ax2.set_xlabel(f'{Ch_name} boxplot of peak values')
    ax2.set_ylabel('Value (AU)')
    
    # Plot histogram for peak widths
    ax3.hist(width_array, color = 'dimgray', alpha = 0.75)
    ax3.set_xlabel(f'{Ch_name} histogram of peak widths')
    ax3.set_ylabel('Occurances')

    # Plot boxplot for peak widths
    bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('dimgray')
    ax4.set_xlabel(f'{Ch_name} boxplot of peak widths')
    ax4.set_ylabel('Peak width (frames)')

    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.close(fig)

    return fig

def return_mean_CCF_figure(
    signal: np.ndarray, 
    shifts: np.ndarray, 
    channel_combo: str, 
    num_frames: int):


    # Plot mean cross-correlation curve with shaded area representing standard deviation
    arr_mean = np.nanmean(signal, axis = 0)
    arr_std = np.nanstd(signal, axis = 0)
    x_axis = np.arange(-num_frames + 1, num_frames)

    # Calculate mean and standard deviation of cross-correlation curves
    fig, ax = plt.subplot_mosaic(mosaic = '''
                                            AA
                                            BC
                                            ''')
    
    # Plot mean cross-correlation curve with shaded area representing standard deviation
    ax['A'].plot(x_axis, arr_mean, color='blue')
    ax['A'].fill_between(x_axis, 
                            arr_mean - arr_std, 
                            arr_mean + arr_std, 
                            color='blue', 
                            alpha=0.2)
    ax['A'].set_title(f'{channel_combo} Mean Crosscorrelation Curve ± Standard Deviation') 

    # Plot histogram of period values
    ax['B'].hist(shifts)
    shifts = [val for val in shifts if not np.isnan(val)]
    ax['B'].set_xlabel(f'Histogram of shift values (frames)')
    ax['B'].set_ylabel('Occurances')

    # Plot boxplot of period values
    ax['C'].boxplot(shifts)
    ax['C'].set_xlabel(f'Boxplot of shift values')
    ax['C'].set_ylabel(f'Measured shift (frames)')

    fig.subplots_adjust(hspace=0.25, wspace=0.5)   
    plt.close(fig)
    return fig