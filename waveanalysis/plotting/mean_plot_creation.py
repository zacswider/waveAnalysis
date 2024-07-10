import numpy as np
import matplotlib.pyplot as plt

def plot_mean_ACF_workflow(
    img_parameters_dict: dict,
    img_props: dict,
    indv_acfs: np.ndarray,
) -> dict:
    '''
    Plot the mean autocorrelation function (ACF) for each channel.

    Args:
        img_parameters_dict (dict): A dictionary containing image parameters.
        img_props (dict): A dictionary containing image properties.
        indv_acfs (np.ndarray): An array of individual autocorrelation functions.

    Returns:
        dict: A dictionary containing the mean ACF figures for each channel.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_frames = img_props['num_frames']
    indv_periods = img_parameters_dict['Period']

    # Initialize dictionary to store the mean ACF figures
    mean_acf_figs = {}

    # Loop through each channel and generate the mean ACF figure
    for channel in range(num_channels):
        # Generate and store the figure for the current channel
        mean_acf_figs[f'Ch{channel + 1} Mean ACF'] = return_mean_ACF_figure(
            signal=indv_acfs[channel], 
            periods=indv_periods[channel], 
            channel=f'Ch{channel + 1}',
            num_frames= num_frames,
            frame_interval=img_props['frame_interval'])    

    return mean_acf_figs

def return_mean_ACF_figure(
    signal: np.ndarray, 
    periods: np.ndarray, 
    channel: str,
    num_frames: int,
    frame_interval: float
) -> plt.Figure:
    '''
    Space saving function to return mean ACF figures
    '''
    # Plot mean autocorrelation curve with shaded area representing standard deviation
    signal_mean = np.nanmean(signal, axis = 0)
    signal_std = np.nanstd(signal, axis = 0)
    x_axis = np.arange(-num_frames + 1, num_frames) * frame_interval

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
    periods = periods[~np.isnan(periods)]
    ax['B'].hist(periods)
    ax['B'].set_xlabel(f'Histogram of period values (seconds)')
    ax['B'].set_ylabel('Occurrences')

    # Plot boxplot of period values
    ax['C'].boxplot(periods)
    ax['C'].set_xlabel(f'Boxplot of period values')
    ax['C'].set_ylabel(f'Measured period (seconds)')

    fig.subplots_adjust(hspace=0.25, wspace=0.5)  
    plt.close(fig)

    return fig

def plot_mean_peak_props_workflow(
    img_parameters_dict: dict,
    img_props: dict
) -> dict:
    '''
    Plot Mean Peak Properties Workflow.

    This function takes in the image parameters dictionary and image properties dictionary
    and returns a dictionary of mean peak property figures for each channel.

    Parameters:
    - img_parameters_dict (dict): A dictionary containing the image parameters for each channel.
    - img_props (dict): A dictionary containing the image properties.

    Returns:
    - mean_peak_figs (dict): A dictionary of mean peak property figures for each channel.
    '''
    # Extract peak properties from the image parameters dictionary
    indv_peak_mins = img_parameters_dict['Peak Min']
    indv_peak_maxs = img_parameters_dict['Peak Max']
    indv_peak_amps = img_parameters_dict['Peak Amp']
    indv_peak_widths = img_parameters_dict['Peak Width']
    indv_peak_offsets = img_parameters_dict['Peak Offset']
    num_channels = img_props['num_channels']

    # Initialize dictionary to store the mean peak property figures
    mean_peak_figs = {}

    # Loop through each channel and generate the mean peak property figure
    for channel in range(num_channels):
        # Generate and store the figure for the current channel
        mean_peak_figs[f'Ch{channel + 1} Peak Props'] = return_mean_prop_peaks_figure(
            min_array=indv_peak_mins[channel], 
            max_array=indv_peak_maxs[channel], 
            amp_array=indv_peak_amps[channel], 
            width_array=indv_peak_widths[channel], 
            offsets_array=indv_peak_offsets[channel],
            Ch_name=f'Ch{channel + 1}')

    return mean_peak_figs

def return_mean_prop_peaks_figure(
    min_array: np.ndarray, 
    max_array: np.ndarray, 
    amp_array: np.ndarray, 
    width_array: np.ndarray,
    offsets_array: np.ndarray,
    Ch_name: str
) -> plt.Figure:
    '''
    Space saving function to return mean peak property figures
    '''
    # Create subplots for histograms and boxplots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    # Filter out NaN values from arrays
    min_array = [val for val in min_array if not np.isnan(val)]
    max_array = [val for val in max_array if not np.isnan(val)]
    amp_array = [val for val in amp_array if not np.isnan(val)]
    width_array = [val for val in width_array if not np.isnan(val)]
    offsets_array = [val for val in offsets_array if not np.isnan(val)]

    # Define plot parameters for histograms and boxplots
    plot_params = { 'amp' : (amp_array, 'tab:blue'),
                    'min' : (min_array, 'tab:purple'),
                    'max' : (max_array, 'tab:orange')
                }
    
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
    ax1.set_ylabel('Occurrences')
    ax2.set_xlabel(f'{Ch_name} boxplot of peak values')
    ax2.set_ylabel('Value (AU)')
    
    # Plot histogram for peak widths
    ax3.hist(width_array, color = 'dimgray', alpha = 0.75)
    ax3.set_xlabel(f'{Ch_name} histogram of peak widths')
    ax3.set_ylabel('Occurrences')

    # Plot boxplot for peak widths
    bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('dimgray')
    ax4.set_xlabel(f'{Ch_name} boxplot of peak widths')
    ax4.set_ylabel('Peak width (seconds)')

    # Plot histogram for peak widths
    ax5.hist(offsets_array, color = 'dimgray', alpha = 0.75)
    ax5.set_xlabel(f'{Ch_name} histogram of peak offsets')
    ax5.set_ylabel('Occurrences')

    # Plot boxplot for peak widths
    bp1 = ax6.boxplot(offsets_array, vert=True, patch_artist=True)
    bp1['boxes'][0].set_facecolor('dimgray')
    ax6.set_xlabel(f'{Ch_name} boxplot of peak offsets')
    ax6.set_ylabel('Peak offset (seconds)')

    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.close(fig)

    return fig

def plot_mean_CCF_workflow(
    img_parameters_dict: dict,
    img_props: dict,
    indv_ccfs: np.ndarray
) -> dict:
    '''
    Plot the mean cross-correlation function (CCF) for each channel combination.

    Args:
        img_parameters_dict (dict): A dictionary containing image parameters.
        img_props (dict): A dictionary containing image properties.
        indv_ccfs (np.ndarray): An array of individual cross-correlation functions.

    Returns:
        dict: A dictionary containing the mean CCF figures for each channel combination.
    '''
    # Extract cross-correlation functions and shifts from the image parameters dictionary
    indv_shifts = img_parameters_dict['Shift']
    channel_combos = img_props['channel_combos']
    num_frames = img_props['num_frames']

    # Initialize dictionary to store the mean CCF figures
    mean_ccf_figs = {}

    # Loop through each channel combination and generate the mean CCF figure
    for combo_number, combo in enumerate(channel_combos):
        # Generate and store the figure for the current channel combination
        mean_ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_mean_CCF_figure(
        signal=indv_ccfs[combo_number], 
        shifts=indv_shifts[combo_number], 
        channel_combo=f'Ch{combo[0] + 1}-Ch{combo[1] + 1}',
        num_frames= num_frames,
        frame_interval=img_props['frame_interval'])

    return mean_ccf_figs

def return_mean_CCF_figure(
    signal: np.ndarray, 
    shifts: np.ndarray, 
    channel_combo: str, 
    num_frames: int,
    frame_interval: float
) -> plt.Figure:
    '''
    Space saving function to return mean CCF figures
    '''
    # Plot mean cross-correlation curve with shaded area representing standard deviation
    arr_mean = np.nanmean(signal, axis = 0)
    arr_std = np.nanstd(signal, axis = 0)
    x_axis = np.arange(-num_frames + 1, num_frames) * frame_interval

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
    ax['B'].set_xlabel(f'Histogram of shift values (seconds)')
    ax['B'].set_ylabel('Occurrences')

    # Plot boxplot of period values
    ax['C'].boxplot(shifts)
    ax['C'].set_xlabel(f'Boxplot of shift values')
    ax['C'].set_ylabel(f'Measured shift (seconds)')

    fig.subplots_adjust(hspace=0.25, wspace=0.5)   
    plt.close(fig)
    
    return fig

def return_mean_wave_speeds_figure(
    wave_speeds: list[float]
) -> plt.Figure:
    '''
    Returns a matplotlib Figure object that contains a histogram and boxplot of wave speeds.

    Parameters:
        wave_speeds (list[float]): A list of wave speeds in µm/s.

    Returns:
        plt.Figure: A matplotlib Figure object containing the histogram and boxplot.

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot histogram of wave speeds
    ax1.hist(wave_speeds, bins = 10, color = 'tab:blue', alpha = 0.75)
    ax1.set_xlabel('Histogram of wave speeds (µm/s)')
    ax1.set_ylabel('Occurrences')
    ax1.set_title('Wave Speeds Histogram')

    # Plot boxplots for peak properties
    boxes = ax2.boxplot(wave_speeds)
    ax2.set_xlabel('Boxplot of wave speeds (µm/s)')
    plt.close(fig)

    return fig