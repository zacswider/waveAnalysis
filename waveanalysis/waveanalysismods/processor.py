import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest

from waveanalysis.image_properties_signal.create_signals import create_standard_signals, create_kymo_signals  
from waveanalysis.signal_processing import calc_indv_ACFs_periods, calc_indv_CCFs_shifts_channelCombos, calc_indv_peak_props
from waveanalysis.plotting import plot_indv_peak_props_workflow, plot_indv_acfs_workflow, plot_indv_ccfs_workflow, save_indv_ccfs_workflow, plot_mean_ACFs_workflow, plot_mean_prop_peaks_workflow

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
        self.xpix, self.ypix  = np.nan, np.nan

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
            self.bin_values, self.total_bins, self.xpix, self.ypix = create_standard_signals(
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
            self.bin_values, self.total_bins = create_kymo_signals(
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
            total_bins=self.total_bins,
            bin_values=self.bin_values,
            analysis_type=self.analysis_type,
            num_submovies=self.num_submovies,
            roll_by=self.roll_by,
            roll_size=self.roll_size,
            xpix=self.xpix,
            ypix=self.ypix
            )
      
        return self.indv_peak_widths, self.indv_peak_maxs, self.indv_peak_mins, self.indv_peak_amps, self.indv_peak_rel_amps, self.indv_peak_props

    def calc_indv_ACFs(self, peak_thresh=0.1):
        
        self.indv_acfs, self.indv_periods = calc_indv_ACFs_periods(
            num_channels=self.num_channels, 
            total_bins=self.total_bins, 
            num_frames=self.num_frames, 
            bin_values=self.bin_values, 
            analysis_type=self.analysis_type, 
            roll_size=self.roll_size, 
            roll_by=self.roll_by, 
            num_submovies=self.num_submovies, 
            xpix=self.xpix, 
            ypix=self.ypix, 
            peak_thresh=peak_thresh
            )

        return self.indv_acfs, self.indv_periods

    def calc_indv_CCFs(self):
        
        self.indv_shifts, self.indv_ccfs, self.channel_combos = calc_indv_CCFs_shifts_channelCombos(
            num_channels=self.num_channels, 
            total_bins=self.total_bins,
            num_frames=self.num_frames, 
            bin_values=self.bin_values, 
            analysis_type=self.analysis_type, 
            roll_size=self.roll_size, 
            roll_by=self.roll_by, 
            num_submovies=self.num_submovies, 
            periods=self.indv_periods
            )

        return self.indv_shifts, self.indv_ccfs, self.channel_combos

############################################
########### INDIVIDUAL BIN PLOTS ###########
############################################
    
    def plot_indv_peak_props(self):
        
        if hasattr(self, 'indv_peak_widths'):
            self.indv_peak_figs = plot_indv_peak_props_workflow(
                num_channels=self.num_channels,
                total_bins=self.total_bins,
                bin_values=self.bin_values,
                analysis_type=self.analysis_type,
                indv_peak_props=self.indv_peak_props
                )

        return self.indv_peak_figs

    def plot_indv_acfs(self):
        
        self.indv_acf_plots = plot_indv_acfs_workflow(
            num_channels=self.num_channels,
            total_bins=self.total_bins,
            bin_values=self.bin_values,
            analysis_type=self.analysis_type,
            acfs=self.indv_acfs,
            periods=self.indv_periods,
            num_frames=self.num_frames
            )
        
        return self.indv_acf_plots

    def plot_indv_ccfs(self):

        if self.num_channels > 1:
            self.indv_ccf_plots = plot_indv_ccfs_workflow(
                total_bins=self.total_bins,
                bin_values=self.bin_values,
                analysis_type=self.analysis_type,
                channel_combos=self.channel_combos,
                indv_shifts=self.indv_shifts,
                indv_ccfs=self.indv_ccfs,
                num_frames=self.num_frames
            )
        
        return self.indv_ccf_plots
    
    def save_indv_ccf_values(self):
        
        self.indv_ccf_values = save_indv_ccfs_workflow(
            indv_ccfs=self.indv_ccfs,
            channel_combos=self.channel_combos,
            bin_values=self.bin_values,
            analysis_type=self.analysis_type,
            total_bins=self.total_bins
        )

        return self.indv_ccf_values
        
############################################
############## MEAN BIN PLOTS ##############
############################################

    def plot_mean_peak_props(self):
        if hasattr(self, 'indv_peak_widths'):
            self.mean_peak_figs = plot_mean_prop_peaks_workflow(
                indv_peak_mins=self.indv_peak_mins,
                indv_peak_maxs=self.indv_peak_maxs,
                indv_peak_amps=self.indv_peak_amps,
                indv_peak_widths=self.indv_peak_widths,
                num_channels=self.num_channels
            )
        
        return self.mean_peak_figs
    
    def plot_mean_ACF(self):
        if hasattr(self, 'indv_acfs'):
            self.mean_acf_figs = plot_mean_ACFs_workflow(
                acfs=self.indv_acfs,
                periods=self.indv_periods,
                num_channels=self.num_channels,
                num_frames=self.num_frames
            )

        return self.mean_acf_figs
    
    def plot_mean_CCF(self):
        """
        This method generates and plots the mean cross-correlation curve with shaded standard deviation area,
        a histogram of shift values, and a boxplot of shift values for each channel combination.

        Returns:
            - dict: The first dictionary contains generated figures of mean CCF plots.
            - dict: The second dictionary contains calculated mean CCF values for each channel combination.
        """
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel_combo: str):
            '''
            Space saving function to generate the plots for the mean CCF plots
            '''
            # Plot mean cross-correlation curve with shaded area representing standard deviation
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_frames + 1, self.num_frames)

            # Calculate mean and standard deviation of cross-correlation curves
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            
            # Plot mean cross-correlation curve with shaded area representing standard deviation
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel_combo} Mean Crosscorrelation Curve ± Standard Deviation') 

            # Plot histogram of period values
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of shift values (frames)')
            ax['B'].set_ylabel('Occurances')

            # Plot boxplot of period values
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of shift values')
            ax['C'].set_ylabel(f'Measured shift (frames)')

            fig.subplots_adjust(hspace=0.25, wspace=0.5)   
            plt.close(fig)
            return fig
        
        def return_mean_CCF_val(arr: np.ndarray):
            '''
            Space saving function to save the values for the mean CCF curves
            '''
            # Calculate mean and standard deviation of cross-correlation curve
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)

            # Combine mean and standard deviation into a list of tuples
            mean_CCF_values = list(zip_longest(range(1, len(arr_mean) + 1), arr_mean, arr_std, fillvalue=None))

            return mean_CCF_values

        # Dictionary to store generated figures and mean CCF values
        self.ccf_figs = {}
        self.mean_ccf_values = {}
                       
        if hasattr(self, 'indv_ccfs'):
            if self.num_channels > 1:
                # Iterate over each channel combination
                for combo_number, combo in enumerate(self.channel_combos):
                    # Generate figure for mean CCF
                    self.ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_figure(self.indv_ccfs[combo_number], 
                                                                                                self.indv_shifts[combo_number], 
                                                                                                f'Ch{combo[0] + 1}-Ch{combo[1] + 1}')
                    # Calculate and store mean CCF values
                    self.mean_ccf_values[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF values.csv'] = return_mean_CCF_val(self.indv_ccfs[combo_number])

        return self.ccf_figs, self.mean_ccf_values
    
    def plot_rolling_summary(self):
        """
        This method plots a rolling summary of the measurements over time.

        Returns:
            - dict: A dictionary containing the generated plots.
        """
        def return_plot(independent_variable, dependent_variable, dependent_error, y_label):    
            '''
            Space saving function to generate the rolling summary plots'''      
            fig, ax = plt.subplots()

            # plot the dataframe
            ax.plot(self.full_movie_summary[independent_variable], 
                         self.full_movie_summary[dependent_variable])
            
            # fill between the ± standard deviation of the dependent variable
            ax.fill_between(x = self.full_movie_summary[independent_variable],
                            y1 = self.full_movie_summary[dependent_variable] - self.full_movie_summary[dependent_error],
                            y2 = self.full_movie_summary[dependent_variable] + self.full_movie_summary[dependent_error],
                            color = 'blue',
                            alpha = 0.25)

            # set axis labels
            ax.set_xlabel('Frame Number')
            ax.set_ylabel(y_label)
            ax.set_title(f'{y_label} over time')
            
            plt.close(fig)
            return fig
        
        # empty dictionary to fill with plots
        self.plot_list = {}

        def add_peak_plots(channel, prop_name):
            '''
            Space saving function
            '''
            self.plot_list[f'Ch {channel} Peak {prop_name}'] = return_plot('Submovie',
                                                                            f'Ch {channel} Mean Peak {prop_name}',
                                                                            f'Ch {channel} StdDev Peak {prop_name}',
                                                                            f'Ch {channel} Mean ± StdDev Peak {prop_name} (frames)')
        
        if hasattr(self, 'indv_periods'):
            for channel in range(self.num_channels):
                self.plot_list[f'Ch {channel + 1} Period'] = return_plot('Submovie',
                                                                          f'Ch {channel + 1} Mean Period',
                                                                          f'Ch {channel + 1} StdDev Period',
                                                                          f'Ch {channel + 1} Mean ± StdDev Period (frames)')
        
        if hasattr(self, 'indv_shifts'):
            for combo_number, combo in enumerate(self.channel_combos):
                self.plot_list[f'Ch{combo[0]+1}-Ch{combo[1]+1} Shift'] = return_plot('Submovie',
                                                                                      f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean Shift',
                                                                                      f'Ch{combo[0]+1}-Ch{combo[1]+1} StdDev Shift',
                                                                                      f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean ± StdDev Shift (frames)')

        if hasattr(self, 'peak_widths'):
            for channel in range(self.num_channels):
                for prop_name in ['Width', 'Max', 'Min', 'Amp']:
                    add_peak_plots(channel + 1, prop_name)

        
        return self.plot_list
    
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
        col_names.extend([f'Box{i}' for i in range(self.total_bins)])
        
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
            
            if hasattr(self, 'peak_widths'):
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
                    pcnt_no_period = (np.count_nonzero(np.isnan(self.indv_periods[submovie, channel])) / self.total_bins) * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(self.indv_periods[submovie, channel])

            if hasattr(self, 'indv_ccfs'):
                for combo_number, combo in enumerate(self.channel_combos):
                    pcnt_no_shift = np.count_nonzero(np.isnan(self.indv_ccfs[submovie, combo_number])) / self.total_bins * 100
                    submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(self.indv_shifts[submovie, combo_number])

            if hasattr(self, 'indv_peak_widths'):
                for channel in range(self.num_channels):
                    # using widths, but because these are all assigned together it applies to all peak properties
                    pcnt_no_peaks = np.count_nonzero(np.isnan(self.indv_peak_widths[submovie, channel])) / self.total_bins * 100
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
    
     
    def summarize_image(self, file_name = None, group_name = None):
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
        self.file_data_summary = {}
        
        if file_name:
            self.file_data_summary['File Name'] = file_name
        if group_name:
            self.file_data_summary['Group Name'] = group_name
        self.file_data_summary['Num Bins'] = self.total_bins

        stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

        if hasattr(self, 'periods_with_stats'):
            pcnt_no_period = [np.count_nonzero(np.isnan(self.indv_periods[channel])) / self.indv_periods[channel].shape[0] * 100 for channel in range(self.num_channels)]
            for channel in range(self.num_channels):
                self.file_data_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period[channel]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch {channel + 1} {stat} Period'] = self.periods_with_stats[channel][ind + 1]
        
        if hasattr(self, 'shifts_with_stats'):
            pcnt_no_shift = [np.count_nonzero(np.isnan(self.indv_shifts[combo_number])) / self.indv_shifts[combo_number].shape[0] * 100 for combo_number, combo in enumerate(self.channel_combos)]
            for combo_number, combo in enumerate(self.channel_combos):
                self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift[combo_number]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = self.shifts_with_stats[combo_number][ind + 1]

        if hasattr(self, 'peak_widths_with_stats'):
            # using widths, but because these are all assigned together it applies to all peak properties
            pcnt_no_peaks = [np.count_nonzero(np.isnan(self.indv_peak_widths[channel])) / self.indv_peak_widths[channel].shape[0] * 100 for channel in range(self.num_channels)]
            for channel in range(self.num_channels):
                self.file_data_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks[channel]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Width'] = self.peak_widths_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Max'] = self.peak_maxs_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Min'] = self.peak_mins_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Amp'] = self.peak_amps_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Rel Amp'] = self.peak_relamp_with_stats[channel][ind + 1]
            
        return self.file_data_summary
    
    def organize_measurements(self):
        """
        This method summarizes measurements statistics by appending them to the beginning of the measurement list
        and returns a pandas DataFrame containing the summarized measurements for each submovie or across all bins.

        Returns:
            - pandas.DataFrame or list of pandas.DataFrame: A DataFrame containing the summarized measurements for each submovie or across all bins.
        """
        def add_stats(measurements: np.ndarray, measurement_name: str):
            '''
            Space saving function to generate the stats for the different channels or channel combos
            '''
            # shift measurements need special treatment to generate the correct measurements and names
            if measurement_name == 'Shift':
                statified = []
                for combo_number, combo in enumerate(self.channel_combos):
                    meas_mean = np.nanmean(measurements[combo_number])
                    meas_median = np.nanmedian(measurements[combo_number])
                    meas_std = np.nanstd(measurements[combo_number])
                    meas_sem = meas_std / np.sqrt(len(measurements[combo_number]))
                    meas_list = list(measurements[combo_number])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(3, meas_sem)
                    meas_list.insert(0, f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {measurement_name}')
                    statified.append(meas_list)

            # acf and peak measurements are just iterated by channel
            else:
                statified = []
                for channel in range(self.num_channels):
                    meas_mean = np.nanmean(measurements[channel])
                    meas_median = np.nanmedian(measurements[channel])
                    meas_std = np.nanstd(measurements[channel])
                    meas_sem = meas_std / np.sqrt(len(measurements[channel]))
                    meas_list = list(measurements[channel])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(3, meas_sem)
                    meas_list.insert(0, f'Ch {channel +1} {measurement_name}')
                    statified.append(meas_list)
            return(statified)

        # column names for the dataframe summarizing the bin results
        col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        col_names.extend([f'Bin {i}' for i in range(self.total_bins)])
    
        # combine all the statified measurements into a single list
        statified_measurements = []

        # insert Mean, Median, StdDev, and SEM into the beginning of each  list
        if hasattr(self, 'indv_acfs'):
            self.periods_with_stats = add_stats(self.indv_periods, 'Period')
            for channel in range(self.num_channels):
                statified_measurements.append(self.periods_with_stats[channel])

        if hasattr(self, 'indv_ccfs'):
            self.shifts_with_stats = add_stats(self.indv_shifts, 'Shift')
            for combo_number, combo in enumerate(self.channel_combos):
                statified_measurements.append(self.shifts_with_stats[combo_number])

        if hasattr(self, 'indv_peak_widths'):
            self.peak_widths_with_stats = add_stats(self.indv_peak_widths, 'Peak Width')
            self.peak_maxs_with_stats = add_stats(self.indv_peak_maxs, 'Peak Max')
            self.peak_mins_with_stats = add_stats(self.indv_peak_mins, 'Peak Min')
            self.peak_amps_with_stats = add_stats(self.indv_peak_amps, 'Peak Amp')
            self.peak_relamp_with_stats = add_stats(self.indv_peak_rel_amps, 'Peak Rel Amp')
            for channel in range(self.num_channels):
                statified_measurements.append(self.peak_widths_with_stats[channel])
                statified_measurements.append(self.peak_maxs_with_stats[channel])
                statified_measurements.append(self.peak_mins_with_stats[channel])
                statified_measurements.append(self.peak_amps_with_stats[channel])
                statified_measurements.append(self.peak_relamp_with_stats[channel])

        self.im_measurements = pd.DataFrame(statified_measurements, columns = col_names)

        return self.im_measurements


    def save_parameter_means_to_csv(self, main_save_path, group_names, summary_df):
        """
        This method saves the means of measurements to CSV files for each channel and metric.

        Parameters:
            - main_save_path (str): The main directory path where the CSV files will be saved.
            - group_names (list of str): A list of group names.
            - summary_df (pandas.DataFrame): A DataFrame containing the summarized measurements.
        """
        for channel in range(self.num_channels):
            # Define data metrics to extract
            metrics_to_extract = [f"Ch {channel + 1} {data}" for data in ['Mean Period', 'Mean Peak Width', 'Mean Peak Max', 'Mean Peak Min', 'Mean Peak Amp', 'Mean Peak Rel Amp']]
            if hasattr(self, 'indv_ccfs'):
                metrics_to_extract = [f"Ch {channel + 1} {data}" for data in ['Mean Period', 'Mean Peak Width', 'Mean Peak Max', 'Mean Peak Min', 'Mean Peak Amp', 'Mean Peak Rel Amp', 'Mean Shift']]
            
            # Create folder for storing results
            folder_path = os.path.join(main_save_path, "!channel_mean_measurements")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Extract data for each group and metric
            result_df = pd.DataFrame(columns=['Data Type', 'Group Name', 'Value'])
            for metric in metrics_to_extract:
                if metric == "Ch 1 Mean Shift" or metric == "Ch 2 Mean Shift":
                    metric = "Ch1-Ch2 Mean Shift"
                for group_name in group_names:
                    group_data = summary_df.loc[summary_df['File Name'].str.contains(group_name)]
                    values = group_data[metric].tolist()
                    result_df = pd.concat([result_df, pd.DataFrame({'Data Type': metric, 'Group Name': group_name, 'Value': values})], ignore_index=True)

            # Save individual tables for each metric
            for metric in metrics_to_extract:
                if metric == "Ch 1 Mean Shift" or metric == "Ch 2 Mean Shift":
                    metric = "Ch1-Ch2 Mean Shift"
                # Define output path for CSV
                output_path = os.path.join(folder_path, f"{metric.lower().replace(' ', '_')}_means.csv")

                # Prepare and sort table 
                metric_table = result_df[result_df['Data Type'] == metric][['Group Name', 'Value']]
                metric_table = pd.pivot_table(metric_table, index=metric_table.index, columns='Group Name', values='Value')
                for col in metric_table.columns:
                    metric_table[col] = sorted(metric_table[col], key=lambda x: 1 if pd.isna(x) or x == '' else 0)
                
                # Save table to CSV
                metric_table.to_csv(output_path, index=False)