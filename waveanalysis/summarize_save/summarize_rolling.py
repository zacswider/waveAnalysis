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
        
    # column names for the dataframe summarizing the box results
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
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
