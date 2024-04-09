import numpy as np
import pandas as pd

def summarize_image(
    img_parameters: dict,
    img_props_dict: dict
) -> pd.DataFrame:
    '''
    Summarizes the image parameters and properties of a standard kymograph.

    Args:
        img_parameters (dict): A dictionary containing the image parameters.
        img_props_dict (dict): A dictionary containing the image properties.

    Returns:
        pd.DataFrame: A dataframe summarizing the bin results.

    '''
    # Extract image properties from the dictionary
    num_bins = img_props_dict['num_bins']
    num_channels = img_props_dict['num_channels']
    channel_combos = img_props_dict['channel_combos']

    # column names for the dataframe summarizing the bin results
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
    col_names.extend([f'Bin {i}' for i in range(num_bins)])

    # combine all the statified measurements into a single list
    im_measurements = []
    parameter_with_stats_dict = {}

    if 'num_submovies' in img_props_dict:
        num_submovies = img_props_dict['num_submovies']

        # insert Mean, Median, StdDev, and SEM into the beginning of each list
        for submovie in range(num_submovies):
            statified_measurements = []
            for parameter, parameter_measurements in img_parameters.items():
                parameter_with_stats = add_stats_for_parameter(parameter_measurements[submovie], parameter, num_channels, channel_combos)
                for channel_combo_stat in parameter_with_stats:
                    statified_measurements.append(channel_combo_stat)

            # create a dataframe from the statified measurements
            submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
            im_measurements.append(submovie_meas_df)
    else:
        # insert Mean, Median, StdDev, and SEM into the beginning of each list
        for parameter, parameter_measurements in img_parameters.items():
            parameter_with_stats = add_stats_for_parameter(parameter_measurements, parameter, num_channels, channel_combos)
            parameter_with_stats_dict[parameter] = parameter_with_stats
            for channel_combo_stat in parameter_with_stats:
                im_measurements.append(channel_combo_stat)

        # create a dataframe from the statified measurements
        im_measurements = pd.DataFrame(im_measurements, columns = col_names)

    return im_measurements, parameter_with_stats_dict

def add_stats_for_parameter(
    measurements: np.ndarray,
    measurement_name: str,
    num_channels: int,
    channel_combos: list = None
) -> list:
    '''
    Calculate statistics for a given measurement parameter.

    Parameters:
        measurements (np.ndarray): Array of measurements.
        measurement_name (str): Name of the measurement parameter.
        num_channels (int): Number of channels.
        channel_combos (list, optional): List of channel combinations. Defaults to None.

    Returns:
        list: List of statistics for the measurement parameter.
    '''
    statified = []

    def calculate_statistics(measurements_subset, channel_label):
        meas_mean = np.nanmean(measurements_subset)
        meas_median = np.nanmedian(measurements_subset)
        meas_std = np.nanstd(measurements_subset)
        meas_sem = meas_std / np.sqrt(len(measurements_subset))
        if isinstance(measurements_subset, np.ndarray):
            measurements_subset = measurements_subset.tolist()
        return [channel_label, meas_mean, meas_median, meas_std, meas_sem] + measurements_subset

    if measurement_name not in ['Wave Speed']:
        for index, item in enumerate(channel_combos if measurement_name in ['Shift', 'Phase Shift'] else range(num_channels)):
            if measurement_name in ['Shift', 'Phase Shift']:
                measurements_subset = measurements[index]
                channel_label = f'Ch{channel_combos[index][0]+1}-Ch{channel_combos[index][1]+1} {measurement_name}'
            else:
                measurements_subset = measurements[item]
                channel_label = f'Ch {item + 1} {measurement_name}'
            
            statified.append(calculate_statistics(measurements_subset, channel_label))

    else:
        measurements = calculate_statistics(measurements, measurement_name)
        statified.append(measurements)
        
    return statified

def combine_stats_for_image_kymo_standard(
    file_name: str, 
    group_name: str,
    img_props: dict,
    img_parameters_dict: dict,
    parameters_with_stats_dict: dict
) -> dict:
    '''
    Combine the statistics for an image in a kymograph or standard analysis.

    Args:
        file_name (str): The name of the file.
        group_name (str): The name of the group.
        img_props (dict): A dictionary containing image properties.
        img_parameters_dict (dict): A dictionary containing image parameters.
        parameters_with_stats_dict (dict): A dictionary containing parameters with statistics.

    Returns:
        dict: A dictionary containing the summarized measurements for each image.
    '''
    # Extract image properties from the dictionary
    num_bins = img_props['num_bins']
    num_channels = img_props['num_channels']
    channel_combos = img_props['channel_combos']

    # dictionary to store the summarized measurements for each image
    file_data_summary = {}
    file_data_summary['File Name'] = file_name if file_name else 'None'
    file_data_summary['Group Name'] = group_name if group_name else 'None'
    file_data_summary['Num Bins'] = num_bins

    # column names for the dataframe summarizing the bin results
    stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

    # Add stats for each Shifts
    if num_channels > 1:
        for combo_number, combo in enumerate(channel_combos):
            shift_data = img_parameters_dict['Shift'][combo_number]
            pcnt_no_shift = np.count_nonzero(np.isnan(shift_data)) / shift_data.shape[0] * 100
            file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = parameters_with_stats_dict['Shift'][combo_number][ind + 1]
                file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Phase Shift'] = parameters_with_stats_dict['Phase Shift'][combo_number][ind + 1]

    # Add stats for each parameter
    for name, measurement in img_parameters_dict.items():
        # Skip the Shift name since it is handled separately
        if name in ['Shift', 'Phase Shift']:
            continue
        # We calculate the number of bins without Period and Peak Amp 
        elif name == 'Period' or name == 'Peak Amp':
            for channel in range(num_channels):
                pcnt_no_parameter = np.count_nonzero(np.isnan(measurement[channel])) / measurement[channel].shape[0] * 100
                param = 'Peaks' if name == 'Peak Amp' else 'Periods'
                file_data_summary[f'Ch {channel + 1} Pcnt No {param}'] = pcnt_no_parameter
                for ind, stat in enumerate(stats_location):
                    file_data_summary[f'Ch {channel + 1} {stat} {name}'] = parameters_with_stats_dict[name][channel][ind + 1]
        # All parameters that are not wave speed
        elif name not in ['Wave Speed']:
            for channel in range(num_channels):        
                for ind, stat in enumerate(stats_location):
                    file_data_summary[f'Ch {channel + 1} {stat} {name}'] = parameters_with_stats_dict[name][channel][ind + 1]
        # Wave Speed is a single value, so it doesn't need to be separated by channel
        elif name in ['Wave Speed']:
            for ind, stat in enumerate(stats_location):
                print(parameters_with_stats_dict[name])
                file_data_summary[f'{stat} {name}'] = parameters_with_stats_dict[name][0][ind + 1]

    return file_data_summary

def combine_stats_rolling(
    img_props_dict: dict,
    img_parameters_dict: dict,
    indv_ccfs: np.ndarray,
) -> pd.DataFrame:
    '''
    Combine statistics for rolling analysis.

    Args:
        img_props_dict (dict): A dictionary containing image properties.
        img_parameters_dict (dict): A dictionary containing image parameters.
        indv_ccfs (np.ndarray): An array containing individual cross-correlation functions.

    Returns:
        pd.DataFrame: A DataFrame containing the combined statistics.

    '''
    # Extract image properties from the dictionary
    num_channels = img_props_dict['num_channels']
    num_bins = img_props_dict['num_bins']
    num_submovies = img_props_dict['num_submovies']
    channel_combos = img_props_dict['channel_combos']

    # Extract image parameters from the dictionary
    indv_periods = img_parameters_dict['Period']
    indv_shifts = img_parameters_dict['Shift']
    indv_peak_widths = img_parameters_dict['Peak Width']

    # Define the statistics to calculate
    stat_name_and_func = {'Mean' : np.nanmean,
                            'Median' : np.nanmedian,
                            'StdDev' : np.nanstd                            
                        }
    
    # Extract image properties from the dictionary
    all_submovie_summary = []
    
    # Loop through each submovie
    for submovie in range(num_submovies):
        # Initialize dictionary to store the summary for the current submovie
        submovie_summary = {'Submovie': submovie + 1}

        # Calculate percentage of no shifts for each channel combination
        if num_channels > 1:
            for combo_number, combo in enumerate(channel_combos):
                pcnt_no_shift = np.count_nonzero(np.isnan(indv_ccfs[submovie, combo_number])) / num_bins * 100
                submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift
                for stat_name, func in stat_name_and_func.items():
                    submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(indv_shifts[submovie, combo_number])

        # Calculate statistics for each channel
        for channel in range(num_channels):
            # Calculate percentage of no periods for the current channel
            pcnt_no_period = (np.count_nonzero(np.isnan(indv_periods[submovie, channel])) / num_bins) * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
            
            # Calculate percentage of no peaks for the current channel
            pcnt_no_peaks = np.count_nonzero(np.isnan(indv_peak_widths[submovie, channel])) / num_bins * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
            
            # Calculate statistics for other parameters excluding Shift and Period
            for name, measurements in img_parameters_dict.items():
                if name != 'Shift':
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} {name}'] = func(measurements[submovie, channel])

        all_submovie_summary.append(submovie_summary)
    
    col_names = [key for key in all_submovie_summary[0].keys()]
    full_movie_summary = pd.DataFrame(all_submovie_summary, columns = col_names)
            
    return full_movie_summary
