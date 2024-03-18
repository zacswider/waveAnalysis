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
    file_data_summary['File Name'] = file_name if file_name else 'None'
    file_data_summary['Group Name'] = group_name if group_name else 'None'
    file_data_summary['Num Bins'] = num_bins

    stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

    if num_channels > 1:
        for combo_number, combo in enumerate(channel_combos):
            shift_data = img_parameters_dict['Shift'][combo_number]
            pcnt_no_shift = np.count_nonzero(np.isnan(shift_data)) / shift_data.shape[0] * 100
            file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = parameters_with_stats_dict['Shift'][combo_number][ind + 1]

    for key, value in img_parameters_dict.items():
        if key == 'Shift':
            continue
        elif key == 'Period' or key == 'Peak Amp':
            for channel in range(num_channels):
                pcnt_no_parameter = np.count_nonzero(np.isnan(img_parameters_dict[key][channel])) / img_parameters_dict[key][channel].shape[0] * 100
                param = 'Peaks' if key == 'Peak Amp' else 'Periods'
                file_data_summary[f'Ch {channel + 1} Pcnt No {param}'] = pcnt_no_parameter
        for channel in range(num_channels):        
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'Ch {channel + 1} {stat} {key}'] = parameters_with_stats_dict[key][channel][ind + 1]

    return file_data_summary
