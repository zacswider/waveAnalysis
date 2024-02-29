import numpy as np
import pandas as pd

def organize_submovie_measurements(
    num_bins: int,
    num_channels: int,
    channel_combos: list,
    num_submovies: int,
    indv_periods: np.ndarray,
    indv_ccfs: np.ndarray,
    indv_peak_widths: np.ndarray,
    indv_peak_maxs: np.ndarray,
    indv_peak_mins: np.ndarray,
    indv_peak_amps: np.ndarray,
    indv_peak_rel_amps: np.ndarray
) -> list:
    """
    This method summarizes measurements statistics by appending them to the beginning of the measurement list
    and returns a list of pandas DataFrames containing the summarized measurements for each submovie.

    Returns:
        - list of pandas.DataFrame: A list of DataFrames containing the summarized measurements for each submovie.
    """
    def add_stats(measurements: np.ndarray, measurement_name: str):
        '''
        Accepts a list of measurements. Calculates the mean, median, standard deviation,
        and append them to the beginning of the list in that order. Finally, appends the name of
        the measurement of the beginning of the list.
        '''

        if measurement_name == 'Shift':
            statified = []
            for combo_number, combo in enumerate(channel_combos):
                meas_mean = np.nanmean(measurements[combo_number])
                meas_median = np.nanmedian(measurements[combo_number])
                meas_std = np.nanstd(measurements[combo_number])
                meas_list = list(measurements[combo_number])
                meas_list.insert(0, meas_mean)
                meas_list.insert(1, meas_median)
                meas_list.insert(2, meas_std)
                meas_list.insert(0, f'Ch{combo[0]+1}-Ch{combo[1]+1} {measurement_name}')
                statified.append(meas_list)

        else:
            statified = []
            for channel in range(num_channels):
                meas_mean = np.nanmean(measurements[channel])
                meas_median = np.nanmedian(measurements[channel])
                meas_std = np.nanstd(measurements[channel])
                meas_list = list(measurements[channel])
                meas_list.insert(0, meas_mean)
                meas_list.insert(1, meas_median)
                meas_list.insert(2, meas_std)
                meas_list.insert(0, f'Ch {channel +1} {measurement_name}')
                statified.append(meas_list)

        return(statified)
        
    # column names for the dataframe summarizing the box results
    col_names = ["Parameter", "Mean", "Median", "StdDev"]
    col_names.extend([f'Box{i}' for i in range(num_bins)])
    
    submovie_measurements = []

    for submovie in range(num_submovies):
        statified_measurements = []

        submovie_periods_with_stats = add_stats(indv_periods[submovie], 'Period')
        for channel in range(num_channels):
            statified_measurements.append(submovie_periods_with_stats[channel])
    
        if num_channels > 1:
            submovie_shifts_with_stats = add_stats(indv_ccfs[submovie], 'Shift')
            for combo_number, _ in enumerate(channel_combos):
                statified_measurements.append(submovie_shifts_with_stats[combo_number])
        
        submovie_widths_with_stats = add_stats(indv_peak_widths[submovie], 'Peak Width')
        submovie_maxs_with_stats = add_stats(indv_peak_maxs[submovie], 'Peak Max')
        submovie_mins_with_stats = add_stats(indv_peak_mins[submovie], 'Peak Min')
        submovie_amps_with_stats = add_stats(indv_peak_amps[submovie], 'Peak Amp')
        submovie_rel_amps_with_stats = add_stats(indv_peak_rel_amps[submovie], 'Peak Rel Amp')
        for channel in range(num_channels):
            statified_measurements.append(submovie_widths_with_stats[channel])
            statified_measurements.append(submovie_maxs_with_stats[channel])
            statified_measurements.append(submovie_mins_with_stats[channel])
            statified_measurements.append(submovie_amps_with_stats[channel])
            statified_measurements.append(submovie_rel_amps_with_stats[channel])

        submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
        submovie_measurements.append(submovie_meas_df)

    return submovie_measurements

def summarize_rolling_file(
    num_bins: int,
    num_channels: int,
    channel_combos: list,
    num_submovies: int,
    indv_periods: np.ndarray,
    indv_shifts: np.ndarray,
    indv_peak_widths: np.ndarray,
    indv_peak_maxs: np.ndarray,
    indv_peak_mins: np.ndarray,
    indv_peak_amps: np.ndarray,
    indv_ccfs: np.ndarray
) -> pd.DataFrame:

    all_submovie_summary = []

    stat_name_and_func = {'Mean' : np.nanmean,
                            'Median' : np.nanmedian,
                            'StdDev' : np.nanstd
                            }

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
