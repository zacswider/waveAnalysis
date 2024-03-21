import numpy as np
import pandas as pd
from .summarize_kymo_standard import add_stats_for_parameter

def summarize_submovie_measurements(
    img_props_dict: dict,
    img_parameters_dict: dict,
) -> list:
    '''
    Summarizes the measurements for each submovie.

    Args:
        img_props_dict (dict): A dictionary containing image properties.
        img_parameters_dict (dict): A dictionary containing image parameters.

    Returns:
        list: A list of dataframes containing the summarized measurements for each submovie.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props_dict['num_channels']
    num_bins = img_props_dict['num_bins']
    num_submovies = img_props_dict['num_submovies']
    channel_combos = img_props_dict['channel_combos']

    # column names for the dataframe summarizing the box results
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
    col_names.extend([f'Box{i}' for i in range(num_bins)])
    
    # combine all the statified measurements into a single list
    submovie_measurements = []

    # insert Mean, Median, StdDev, and SEM into the beginning of each list
    for submovie in range(num_submovies):
        statified_measurements = []
        for key, value in img_parameters_dict.items():
            # Skip the Shift key since it is handled separately
            if key == 'Shift':
                continue
            else:
                # Add stats for each channel
                parameter_with_stats = add_stats_for_parameter(img_parameters_dict[key][submovie], key, num_channels, channel_combos)
                for channel in range(num_channels):
                    statified_measurements.append(parameter_with_stats[channel])
                    # TODO: need to do separate for wave speed
        
        # Add stats for Shift
        if num_channels > 1:
            shifts_with_stats = add_stats_for_parameter(img_parameters_dict['Shift'][submovie], 'Shift', num_channels, channel_combos)
            for combo_number, combo in enumerate(channel_combos):
                statified_measurements.append(shifts_with_stats[combo_number])

        submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
        submovie_measurements.append(submovie_meas_df)

    return submovie_measurements

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
    all_submovie_summary = []

    # Define the statistics to calculate
    stat_name_and_func = {'Mean' : np.nanmean,
                            'Median' : np.nanmedian,
                            'StdDev' : np.nanstd                            
                        }
    
    # Extract image properties from the dictionary
    num_channels = img_props_dict['num_channels']
    num_bins = img_props_dict['num_bins']
    num_submovies = img_props_dict['num_submovies']
    channel_combos = img_props_dict['channel_combos']

    # Extract image parameters from the dictionary
    indv_periods = img_parameters_dict['Period']
    indv_shifts = img_parameters_dict['Shift']
    indv_peak_widths = img_parameters_dict['Peak Width']

    # Loop through each submovie
    for submovie in range(num_submovies):
        # Initialize dictionary to store the summary for the current submovie
        submovie_summary = {}
        submovie_summary['Submovie'] = submovie + 1 
        
        # Calculate the percentage of no periods for each channel
        for channel in range(num_channels):
            pcnt_no_period = (np.count_nonzero(np.isnan(indv_periods[submovie, channel])) / num_bins) * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
            for stat_name, func in stat_name_and_func.items():
                submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(indv_periods[submovie, channel])

        # Calculate the percentage of no shifts for each channel combination
        if num_channels > 1:
            for combo_number, combo in enumerate(channel_combos):
                pcnt_no_shift = np.count_nonzero(np.isnan(indv_ccfs[submovie, combo_number])) / num_bins * 100
                submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                for stat_name, func in stat_name_and_func.items():
                    submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(indv_shifts[submovie, combo_number])

        # Calculate the percentage of no peaks for each channel
        for channel in range(num_channels):
            # using widths, but because these are all assigned together it applies to all peak properties
            pcnt_no_peaks = np.count_nonzero(np.isnan(indv_peak_widths[submovie, channel])) / num_bins * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
            for name, measurements in img_parameters_dict.items():
                # Skip the Shift key since it is handled separately
                if name == 'Shift' or name == 'Period':
                    continue
                else:
                    # Add stats for each channel
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} {name}'] = func(measurements[submovie, channel])
                         # TODO: need to do separate for wave speed

        all_submovie_summary.append(submovie_summary)
    
    col_names = [key for key in all_submovie_summary[0].keys()]
    full_movie_summary = pd.DataFrame(all_submovie_summary, columns = col_names)
            
    return full_movie_summary
