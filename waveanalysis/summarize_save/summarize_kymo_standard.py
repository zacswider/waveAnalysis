import numpy as np
import pandas as pd

def summarize_image_standard_kymo(
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
        elif key == 'Wave Speed':
            parameter_with_stats = add_stats_for_parameter(img_parameters[key], key, num_channels, channel_combos)
            parameter_with_stats = sum(parameter_with_stats, []) # flatten the list
            parameter_with_stats_dict[key] = parameter_with_stats
            statified_measurements.append(parameter_with_stats)
        else:
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

def add_stats_for_parameter(
        measurements: np.ndarray, 
        measurement_name: str, 
        num_channels: int, 
        channel_combos: list = None
) -> list:
        
        statified = []

        if measurement_name != 'Wave Speed':
            for index, item in enumerate(channel_combos if measurement_name == 'Shift' else range(num_channels)):
                if measurement_name == 'Shift':
                    measurements_subset = measurements[index]
                    channel_label = f'Ch{channel_combos[index][0]+1}-Ch{channel_combos[index][1]+1} {measurement_name}'
                else:
                    measurements_subset = measurements[item]
                    channel_label = f'Ch {item + 1} {measurement_name}'
                
                meas_mean = np.nanmean(measurements_subset)
                meas_median = np.nanmedian(measurements_subset)
                meas_std = np.nanstd(measurements_subset)
                meas_sem = meas_std / np.sqrt(len(measurements_subset))
                meas_list = [channel_label, meas_mean, meas_median, meas_std, meas_sem]
                meas_list.extend(measurements_subset.tolist())
                statified.append(meas_list)

        else:
            meas_mean = np.nanmean(measurements)
            meas_median = np.nanmedian(measurements)
            meas_std = np.nanstd(measurements)
            meas_sem = meas_std / np.sqrt(len(measurements))
            meas_list = [measurement_name, meas_mean, meas_median, meas_std, meas_sem]
            meas_list.extend(measurements)
            statified.append(meas_list)
            
        return statified

def combine_stats_for_image_kymo_standard(
    file_name: str, 
    group_name: str,
    img_props: dict,
    img_parameters_dict: dict,
    parameters_with_stats_dict: dict
    ) -> dict:

    num_bins = img_props['num_bins']
    num_channels = img_props['num_channels']
    channel_combos = img_props['channel_combos']

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
        elif key != 'Wave Speed':
            for channel in range(num_channels):        
                for ind, stat in enumerate(stats_location):
                    file_data_summary[f'Ch {channel + 1} {stat} {key}'] = parameters_with_stats_dict[key][channel][ind + 1]
        elif key == 'Wave Speed':
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'{stat} {key}'] = parameters_with_stats_dict[key][ind + 1]

    return file_data_summary
