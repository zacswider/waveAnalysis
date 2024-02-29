import tifffile
import numpy as np
import pandas as pd

from waveanalysis.image_properties_signal.create_np_arrays import create_array_from_standard_rolling, create_array_from_kymo  
from waveanalysis.signal_processing import calc_indv_ACFs_periods, calc_indv_CCFs_shifts_channelCombos, calc_indv_peak_props
from waveanalysis.plotting import plot_indv_peak_props_workflow, plot_indv_acfs_workflow, plot_indv_ccfs_workflow, save_indv_ccfs_workflow, plot_mean_ACFs_workflow, plot_mean_prop_peaks_workflow, plot_mean_CCFs_workflow, save_mean_CCF_values_workflow, plot_rolling_mean_periods, plot_rolling_mean_shifts, plot_rolling_mean_peak_props

np.seterr(divide='ignore', invalid='ignore')

# TODO: remove all of these functions out of the class and into the main workflow file. 
# TODO: change all functions such that they take in less than 5 parameters, and are less than 20 lines of code.

class TotalSignalProcessor:
    def __init__(self, analysis_type, image_path, image, kern=None, step=None, roll_size=None, roll_by=None, line_width=None):
        # Import variables
        self.analysis_type = analysis_type
        self.line_width = line_width
        self.roll_size = roll_size
        self.roll_by = roll_by
        self.kernel_size = kern
        self.step = step

        # save other values to np.nan for now
        self.num_submovies = np.nan
        self.num_x_bins, self.num_y_bins  = np.nan, np.nan

        # Import image and extract metadata
        self.image_path = image_path
        self.image = image
        with tifffile.TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)

        if self.analysis_type == "standard" or self.analysis_type == "rolling":
            self.num_frames = metadata.get('frames', 1)
            
        if self.analysis_type == "rolling":
            assert isinstance(self.roll_size, int) and isinstance(self.roll_by, int), 'Roll size and roll by must be integers'
            self.num_submovies = (self.num_frames - self.roll_size) // self.roll_by
            
        if self.analysis_type == "standard" or self.analysis_type == "rolling":
            self.bin_values, self.num_bins, self.num_x_bins, self.num_y_bins = create_array_from_standard_rolling(
                kernel_size=self.kernel_size, 
                step=self.step, 
                num_channels=self.num_channels, 
                num_frames=self.num_frames, 
                image=self.image
                )

        # Use lines for kymograph analysis
        else:
            self.total_columns = self.image.shape[-1]
            self.num_frames = self.image.shape[-2]
            self.bin_values, self.num_bins = create_array_from_kymo(
                line_width=self.line_width, 
                total_columns=self.total_columns, 
                step=self.step, 
                num_channels=self.num_channels, 
                num_frames=self.num_frames, 
                image=self.image
                )
        
############################################
######## INDIVIDUAL BIN CALCULATION ########
############################################
    
    def calc_indv_peak_props(self):
       
        self.indv_peak_widths, self.indv_peak_maxs, self.indv_peak_mins, self.indv_peak_amps, self.indv_peak_rel_amps, self.indv_peak_props = calc_indv_peak_props(
            num_channels=self.num_channels,
            num_bins=self.num_bins,
            bin_values=self.bin_values,
            analysis_type=self.analysis_type,
            num_submovies=self.num_submovies,
            roll_by=self.roll_by,
            roll_size=self.roll_size,
            num_x_bins=self.num_x_bins,
            num_y_bins=self.num_y_bins
            )
      
        return self.indv_peak_widths, self.indv_peak_maxs, self.indv_peak_mins, self.indv_peak_amps, self.indv_peak_rel_amps, self.indv_peak_props

    def calc_indv_ACFs(self, peak_thresh=0.1):
        
        self.indv_acfs, self.indv_periods = calc_indv_ACFs_periods(
            num_channels=self.num_channels, 
            num_bins=self.num_bins, 
            num_frames=self.num_frames, 
            bin_values=self.bin_values, 
            analysis_type=self.analysis_type, 
            roll_size=self.roll_size, 
            roll_by=self.roll_by, 
            num_submovies=self.num_submovies, 
            num_x_bins=self.num_x_bins, 
            num_y_bins=self.num_y_bins, 
            peak_thresh=peak_thresh
            )

        return self.indv_acfs, self.indv_periods

    def calc_indv_CCFs(self):
        
        self.indv_shifts, self.indv_ccfs, self.channel_combos = calc_indv_CCFs_shifts_channelCombos(
            num_channels=self.num_channels, 
            num_bins=self.num_bins,
            num_frames=self.num_frames, 
            bin_values=self.bin_values, 
            analysis_type=self.analysis_type, 
            roll_size=self.roll_size, 
            roll_by=self.roll_by, 
            num_submovies=self.num_submovies, 
            periods=self.indv_periods
            )

        return self.indv_shifts, self.indv_ccfs, self.channel_combos








    def plot_rolling_summary(self):        

        self.rolling_mean_plots_dict = {}

        if hasattr(self, 'indv_periods'):
            self.rolling_mean_period_plots = plot_rolling_mean_periods(
                num_channels=self.num_channels,
                fullmovie_summary=self.full_movie_summary
            )

            self.rolling_mean_plots_dict.update(self.rolling_mean_period_plots)
        
        if hasattr(self, 'indv_shifts'):
            self.rolling_mean_shifts_plots = (plot_rolling_mean_shifts(
                channel_combos=self.channel_combos,
                fullmovie_summary=self.full_movie_summary
            ))

            self.rolling_mean_plots_dict.update(self.rolling_mean_shifts_plots)


        if hasattr(self, 'indv_peak_widths'):
            self.rolling_mean_prop_plots = (plot_rolling_mean_peak_props(
                num_channels=self.num_channels,
                fullmovie_summary=self.full_movie_summary
            ))

            self.rolling_mean_plots_dict.update(self.rolling_mean_prop_plots)
            
        return self.rolling_mean_plots_dict
    



############################################
############ DATA ORGANIZATION #############
############################################ 

    def get_submovie_measurements(self):
        '''
        Gathers period, shift, and peak properties measurements (if they exist), appends some simple statistics, 
        and returns a SEPARATE dataframe with raw and summarized measurements for each submovie in the dataset.
        returns:
        self.submovie_measurements is a list of dataframes, one for each submovie in the full sequence
        '''
        
        # function to summarize measurements statistics by appending them to the beginning of the measurement list
        def add_stats(measurements: np.ndarray, measurement_name: str):
            '''
            Accepts a list of measurements. Calculates the mean, median, standard deviation,
            and append them to the beginning of the list in that order. Finally, appends the name of
            the measurement of the beginning of the list.
            '''

            if measurement_name == 'Shift':
                statified = []
                for combo_number, combo in enumerate(self.channel_combos):
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
                for channel in range(self.num_channels):
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
        col_names.extend([f'Box{i}' for i in range(self.num_bins)])
        
        self.submovie_measurements = []

        for submovie in range(self.num_submovies):
            statified_measurements = []

            if hasattr(self, 'indv_acfs'):
                submovie_periods_with_stats = add_stats(self.indv_periods[submovie], 'Period')
                for channel in range(self.num_channels):
                    statified_measurements.append(submovie_periods_with_stats[channel])
            
            if hasattr(self, 'indv_ccfs'):
                submovie_shifts_with_stats = add_stats(self.indv_ccfs[submovie], 'Shift')
                for combo_number, _ in enumerate(self.channel_combos):
                    statified_measurements.append(submovie_shifts_with_stats[combo_number])
            
            if hasattr(self, 'indv_peak_widths'):
                submovie_widths_with_stats = add_stats(self.indv_peak_widths[submovie], 'Peak Width')
                submovie_maxs_with_stats = add_stats(self.indv_peak_maxs[submovie], 'Peak Max')
                submovie_mins_with_stats = add_stats(self.indv_peak_mins[submovie], 'Peak Min')
                submovie_amps_with_stats = add_stats(self.indv_peak_amps[submovie], 'Peak Amp')
                submovie_rel_amps_with_stats = add_stats(self.indv_peak_rel_amps[submovie], 'Peak Rel Amp')
                for channel in range(self.num_channels):
                    statified_measurements.append(submovie_widths_with_stats[channel])
                    statified_measurements.append(submovie_maxs_with_stats[channel])
                    statified_measurements.append(submovie_mins_with_stats[channel])
                    statified_measurements.append(submovie_amps_with_stats[channel])
                    statified_measurements.append(submovie_rel_amps_with_stats[channel])

            submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
            self.submovie_measurements.append(submovie_meas_df)

        return self.submovie_measurements 


    def summarize_rolling_file(self):
        """
        This method calculates and summarizes various measurements for each submovie in a rolling analysis, including
        statistics on periods, shifts, and peak properties. It returns a pandas DataFrame containing the summarized measurements.

        Returns:
            - pandas.DataFrame: A DataFrame containing the summarized measurements for each submovie in the rolling analysis.
        """
        all_submovie_summary = []

        stat_name_and_func = {'Mean' : np.nanmean,
                              'Median' : np.nanmedian,
                              'StdDev' : np.nanstd
                              }

        for submovie in range(self.num_submovies):
            submovie_summary = {}
            submovie_summary['Submovie'] = submovie + 1 
            
            if hasattr(self, 'indv_acfs'):
                for channel in range(self.num_channels):
                    pcnt_no_period = (np.count_nonzero(np.isnan(self.indv_periods[submovie, channel])) / self.num_bins) * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(self.indv_periods[submovie, channel])

            if hasattr(self, 'indv_ccfs'):
                for combo_number, combo in enumerate(self.channel_combos):
                    pcnt_no_shift = np.count_nonzero(np.isnan(self.indv_ccfs[submovie, combo_number])) / self.num_bins * 100
                    submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(self.indv_shifts[submovie, combo_number])

            if hasattr(self, 'indv_peak_widths'):
                for channel in range(self.num_channels):
                    # using widths, but because these are all assigned together it applies to all peak properties
                    pcnt_no_peaks = np.count_nonzero(np.isnan(self.indv_peak_widths[submovie, channel])) / self.num_bins * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Width'] = func(self.indv_peak_widths[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Max'] = func(self.indv_peak_maxs[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Min'] = func(self.indv_peak_mins[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Amp'] = func(self.indv_peak_amps[submovie, channel])
            all_submovie_summary.append(submovie_summary)
        
        col_names = [key for key in all_submovie_summary[0].keys()]
        self.full_movie_summary = pd.DataFrame(all_submovie_summary, columns = col_names)
                
        return self.full_movie_summary

