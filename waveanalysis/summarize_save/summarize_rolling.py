import numpy as np
import pandas as pd
from .summarize_kymo_standard import add_stats_for_parameter

def summarize_submovie_measurements(
    img_props_dict: dict,
    img_parameters_dict: dict,
) -> list:
    
    num_channels = img_props_dict['num_channels']
    num_bins = img_props_dict['num_bins']
    num_submovies = img_props_dict['num_submovies']
    channel_combos = img_props_dict['channel_combos']

    # column names for the dataframe summarizing the box results
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
    col_names.extend([f'Box{i}' for i in range(num_bins)])
    
    submovie_measurements = []

    for submovie in range(num_submovies):
        statified_measurements = []
        for key, value in img_parameters_dict.items():
            if key == 'Shift':
                continue
            else:
                parameter_with_stats = add_stats_for_parameter(img_parameters_dict[key][submovie], key, num_channels, channel_combos)
                for channel in range(num_channels):
                    statified_measurements.append(parameter_with_stats[channel])
                
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

    all_submovie_summary = []

    stat_name_and_func = {'Mean' : np.nanmean,
                            'Median' : np.nanmedian,
                            'StdDev' : np.nanstd
                            }
    
    num_channels = img_props_dict['num_channels']
    num_bins = img_props_dict['num_bins']
    num_submovies = img_props_dict['num_submovies']
    channel_combos = img_props_dict['channel_combos']

    indv_periods = img_parameters_dict['Period']
    indv_shifts = img_parameters_dict['Shift']
    indv_peak_widths = img_parameters_dict['Peak Width']
    indv_peak_maxs = img_parameters_dict['Peak Max']
    indv_peak_mins = img_parameters_dict['Peak Min']
    indv_peak_amps = img_parameters_dict['Peak Amp']

    for submovie in range(num_submovies):
        submovie_summary = {}
        submovie_summary['Submovie'] = submovie + 1 
        

        for channel in range(num_channels):
            pcnt_no_period = (np.count_nonzero(np.isnan(indv_periods[submovie, channel])) / num_bins) * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
            for stat_name, func in stat_name_and_func.items():
                submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(indv_periods[submovie, channel])

        if num_channels > 1:
            for combo_number, combo in enumerate(channel_combos):
                pcnt_no_shift = np.count_nonzero(np.isnan(indv_ccfs[submovie, combo_number])) / num_bins * 100
                submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                for stat_name, func in stat_name_and_func.items():
                    submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(indv_shifts[submovie, combo_number])

    
        for channel in range(num_channels):
            # using widths, but because these are all assigned together it applies to all peak properties
            pcnt_no_peaks = np.count_nonzero(np.isnan(indv_peak_widths[submovie, channel])) / num_bins * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
            for stat_name, func in stat_name_and_func.items():
                submovie_summary[f'Ch {channel + 1} {stat_name} Peak Width'] = func(indv_peak_widths[submovie, channel])
                submovie_summary[f'Ch {channel + 1} {stat_name} Peak Max'] = func(indv_peak_maxs[submovie, channel])
                submovie_summary[f'Ch {channel + 1} {stat_name} Peak Min'] = func(indv_peak_mins[submovie, channel])
                submovie_summary[f'Ch {channel + 1} {stat_name} Peak Amp'] = func(indv_peak_amps[submovie, channel])
        
        all_submovie_summary.append(submovie_summary)
    
    col_names = [key for key in all_submovie_summary[0].keys()]
    full_movie_summary = pd.DataFrame(all_submovie_summary, columns = col_names)
            
    return full_movie_summary
