import numpy as np
import pandas as pd

def organize_standard_kymo_measurements_for_file(
    num_bins: int,
    num_channels: int,
    channel_combos: list,
    indv_periods: np.ndarray,
    indv_shifts: np.ndarray,
    indv_peak_widths: np.ndarray,
    indv_peak_maxs: np.ndarray,
    indv_peak_mins: np.ndarray,
    indv_peak_amps: np.ndarray,
    indv_peak_rel_amps: np.ndarray
) -> pd.DataFrame:
    """
    This method summarizes measurements statistics by appending them to the beginning of the measurement list
    and returns a pandas DataFrame containing the summarized measurements for each submovie or across all bins.

    Returns:
        - pandas.DataFrame or list of pandas.DataFrame: A DataFrame containing the summarized measurements for each submovie or across all bins.
    """
    # TODO: add this function to new file and use for both rolling and kymo
    
    def add_stats(measurements: np.ndarray, measurement_name: str):
        statified = []
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
        
        return statified

    # column names for the dataframe summarizing the bin results
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
    col_names.extend([f'Bin {i}' for i in range(num_bins)])

    # combine all the statified measurements into a single list
    statified_measurements = []

    # insert Mean, Median, StdDev, and SEM into the beginning of each  list
    periods_with_stats = add_stats(indv_periods, 'Period')
    for channel in range(num_channels):
        statified_measurements.append(periods_with_stats[channel])

    if num_channels > 1:
        shifts_with_stats = add_stats(indv_shifts, 'Shift')
        for combo_number, combo in enumerate(channel_combos):
            statified_measurements.append(shifts_with_stats[combo_number])

    # peak props
    peak_widths_with_stats = add_stats(indv_peak_widths, 'Peak Width')
    peak_maxs_with_stats = add_stats(indv_peak_maxs, 'Peak Max')
    peak_mins_with_stats = add_stats(indv_peak_mins, 'Peak Min')
    peak_amps_with_stats = add_stats(indv_peak_amps, 'Peak Amp')
    peak_relamp_with_stats = add_stats(indv_peak_rel_amps, 'Peak Rel Amp')
    for channel in range(num_channels):
        statified_measurements.append(peak_widths_with_stats[channel])
        statified_measurements.append(peak_maxs_with_stats[channel])
        statified_measurements.append(peak_mins_with_stats[channel])
        statified_measurements.append(peak_amps_with_stats[channel])
        statified_measurements.append(peak_relamp_with_stats[channel])

    im_measurements = pd.DataFrame(statified_measurements, columns = col_names)

    return im_measurements, periods_with_stats, shifts_with_stats, peak_widths_with_stats, peak_maxs_with_stats, peak_mins_with_stats, peak_amps_with_stats, peak_relamp_with_stats


def summarize_standard_kymo_measurements_for_file(
    file_name: str, 
    group_name: str,
    num_bins: int,
    num_channels: int,
    channel_combos: list,
    indv_periods: np.ndarray,
    periods_with_stats: list,
    indv_shifts: np.ndarray,
    shifts_with_stats: list,
    indv_peak_widths: np.ndarray,
    peak_widths_with_stats: list,
    peak_maxs_with_stats: list,
    peak_mins_with_stats: list,
    peak_amps_with_stats: list,
    peak_relamp_with_stats: list,
    ) -> dict:
    """
    This method calculates and summarizes various measurements for each image, including statistics on periods,
    shifts, and peak properties. It returns a dictionary containing the summarized measurements.

    Parameters:
        - file_name (str, optional): The name of the file.
        - group_name (str, optional): The name of the group to which the image belongs.

    Returns:
        - dict: A dictionary containing the summarized measurements for each image.
    """
    # dictionary to store the summarized measurements for each image
    file_data_summary = {}
    
    if file_name:
        file_data_summary['File Name'] = file_name
    if group_name:
        file_data_summary['Group Name'] = group_name
    file_data_summary['Num Bins'] = num_bins

    stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

    pcnt_no_period = [np.count_nonzero(np.isnan(indv_periods[channel])) / indv_periods[channel].shape[0] * 100 for channel in range(num_channels)]
    for channel in range(num_channels):
        file_data_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period[channel]
        for ind, stat in enumerate(stats_location):
            file_data_summary[f'Ch {channel + 1} {stat} Period'] = periods_with_stats[channel][ind + 1]
    
    if num_channels > 1:
        pcnt_no_shift = [np.count_nonzero(np.isnan(indv_shifts[combo_number])) / indv_shifts[combo_number].shape[0] * 100 for combo_number, combo in enumerate(channel_combos)]
        for combo_number, combo in enumerate(channel_combos):
            file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift[combo_number]
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = shifts_with_stats[combo_number][ind + 1]

    # using widths, but because these are all assigned together it applies to all peak properties
    pcnt_no_peaks = [np.count_nonzero(np.isnan(indv_peak_widths[channel])) / indv_peak_widths[channel].shape[0] * 100 for channel in range(num_channels)]
    for channel in range(num_channels):
        file_data_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks[channel]
        for ind, stat in enumerate(stats_location):
            file_data_summary[f'Ch {channel + 1} {stat} Peak Width'] = peak_widths_with_stats[channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Max'] = peak_maxs_with_stats[channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Min'] = peak_mins_with_stats[channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Amp'] = peak_amps_with_stats[channel][ind + 1]
            file_data_summary[f'Ch {channel + 1} {stat} Peak Rel Amp'] = peak_relamp_with_stats[channel][ind + 1]
    
    return file_data_summary