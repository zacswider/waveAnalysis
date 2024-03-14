import numpy as np
import pandas as pd
from .add_stats import add_stats_for_parameter

def organize_standard_kymo_measurements_for_file(
    num_bins: int,
    num_channels: int,
    channel_combos: list,
    img_parameters: dict,
) -> pd.DataFrame:
    # column names for the dataframe summarizing the bin results
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
    col_names.extend([f'Bin {i}' for i in range(num_bins)])

    # combine all the statified measurements into a single list
    statified_measurements = []
    parameter_with_stats_dict = {}

    # insert Mean, Median, StdDev, and SEM into the beginning of each list
    for key, value in img_parameters.items():
        if key == 'Shift':
            continue
        parameter_with_stats = add_stats_for_parameter(img_parameters[key], key, num_channels, channel_combos)
        parameter_with_stats_dict[key] = parameter_with_stats
        for channel in range(num_channels):
            statified_measurements.append(parameter_with_stats[channel])
            
    if num_channels > 1:
        shifts_with_stats = add_stats_for_parameter(img_parameters['Shift'], 'Shift', num_channels, channel_combos)
        parameter_with_stats_dict['Shift'] = shifts_with_stats
        for combo_number, combo in enumerate(channel_combos):
            statified_measurements.append(shifts_with_stats[combo_number])

    im_measurements = pd.DataFrame(statified_measurements, columns = col_names)

    return im_measurements, parameter_with_stats_dict


def summarize_standard_kymo_measurements_for_file(
    file_name: str, 
    group_name: str,
    num_bins: int,
    num_channels: int,
    channel_combos: list,
    img_parameters_dict: dict,
    parameters_with_stats_dict: dict
    ) -> dict:
    # dictionary to store the summarized measurements for each image
    file_data_summary = {}
    
    if file_name:
        file_data_summary['File Name'] = file_name
    if group_name:
        file_data_summary['Group Name'] = group_name
    file_data_summary['Num Bins'] = num_bins

    stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

    pcnt_no_period = [np.count_nonzero(np.isnan(img_parameters_dict['Period'][channel])) / img_parameters_dict['Period'][channel].shape[0] * 100 for channel in range(num_channels)]
    for channel in range(num_channels):
        file_data_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period[channel]
        for ind, stat in enumerate(stats_location):
            file_data_summary[f'Ch {channel + 1} {stat} Period'] = parameters_with_stats_dict['Period'][channel][ind + 1]

    if num_channels > 1:
        pcnt_no_shift = [np.count_nonzero(np.isnan(img_parameters_dict['Shift'][combo_number])) / img_parameters_dict['Shift'][combo_number].shape[0] * 100 for combo_number, combo in enumerate(channel_combos)]
        for combo_number, combo in enumerate(channel_combos):
            file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift[combo_number]
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = parameters_with_stats_dict['Shift'][combo_number][ind + 1]

    pcnt_no_peaks = [np.count_nonzero(np.isnan(img_parameters_dict['Peak Width'][channel])) / img_parameters_dict['Peak Width'][channel].shape[0] * 100 for channel in range(num_channels)]
    for channel in range(num_channels):
        file_data_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks[channel]
        for ind, stat in enumerate(stats_location):
            file_data_summary[f'Ch {channel + 1} {stat} Peak Width'] = parameters_with_stats_dict['Peak Width'][channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Max'] = parameters_with_stats_dict['Peak Max'][channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Min'] = parameters_with_stats_dict['Peak Min'][channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Amp'] = parameters_with_stats_dict['Peak Amp'][channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Rel Amp'] = parameters_with_stats_dict['Peak Rel Amp'][channel][ind + 1]
            
    return file_data_summary
