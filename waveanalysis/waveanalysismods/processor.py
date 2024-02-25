import os
import csv
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal as sig
import matplotlib.pyplot as plt
from itertools import zip_longest

from waveanalysis.image_properties_signal.create_signals import create_standard_signals, create_kymo_signals  
from waveanalysis.signal_processing import calc_indv_ACFs_periods, calc_indv_CCFs_shifts_channelCombos

np.seterr(divide='ignore', invalid='ignore')

class TotalSignalProcessor:
    def __init__(self, analysis_type, image_path, image, kern=None, step=None, roll_size=None, roll_by=None, line_width=None):
        # Import variables
        self.analysis_type = analysis_type
        self.line_width = line_width
        self.roll_size = roll_size
        self.roll_by = roll_by
        self.kernel_size = kern
        self.step = step
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
            self.bin_values, self.total_bins, self.xpix, self.ypix = create_standard_signals(kernel_size=self.kernel_size, step=self.step, num_channels=self.num_channels, num_frames=self.num_frames, image=self.image)

        # Use lines for kymograph analysis
        else:
            self.total_columns = self.image.shape[-1]
            self.num_frames = self.image.shape[-2]
            self.bin_values, self.total_bins = create_kymo_signals(line_width=self.line_width, total_columns=self.total_columns, step=self.step, num_channels=self.num_channels, num_frames=self.num_frames, image=self.image)
        
############################################
######## INDIVIDUAL BIN CALCULATION ########
############################################
    
    def calc_indv_peak_props(self):
        """
        This method computes various peak properties for each channel and bin of the analyzed data.

        Returns:
            - ind_peak_widths (numpy.ndarray): Array of peak widths.
            - ind_peak_maxs (numpy.ndarray): Array of peak maximum values.
            - ind_peak_mins (numpy.ndarray): Array of peak minimum values.
            - ind_peak_amps (numpy.ndarray): Array of peak amplitudes.
            - ind_peak_rel_amps (numpy.ndarray): Array of relative peak amplitudes.
            - ind_peak_props (dict): Dictionary containing additional peak properties.
        """
        def indv_props(channel, signal, bin, submovie = None):
            """
            This function calculates various peak properties for a given signal.

            Parameters:
                - signal (numpy.ndarray): Input signal.
                - bin (int): Index of the bin.
                - submovie (int): Index of the submovie. Defaults to None.
            """
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

            # If peaks detected, calculate properties, otherwise return NaNs
            if len(peaks) > 0:
                proms, _, _ = sig.peak_prominences(signal, peaks)
                widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
                mean_width = np.mean(widths, axis=0)
                mean_max = np.mean(signal[peaks], axis = 0)
                mean_min = np.mean(signal[peaks]-proms, axis = 0)
            else:
                mean_width = np.nan
                mean_max = np.nan
                mean_min = np.nan
                peaks = np.nan
                proms = np.nan 
                heights = np.nan
                leftIndex = np.nan
                rightIndex = np.nan

            # If rolling analysis
            if submovie != None:
                # Store peak measurements for each bin in each channel of a submovie
                self.ind_peak_widths[submovie, channel, bin] = mean_width
                self.ind_peak_maxs[submovie, channel, bin] = mean_max
                self.ind_peak_mins[submovie, channel, bin] = mean_min
            
            else:
                # Store peak measurements for each bin in each channel
                self.ind_peak_widths[channel, bin] = mean_width
                self.ind_peak_maxs[channel, bin] = mean_max
                self.ind_peak_mins[channel, bin] = mean_min
                self.ind_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': signal, 
                                                        'peaks': peaks,
                                                        'proms': proms, 
                                                        'heights': heights, 
                                                        'leftIndex': leftIndex, 
                                                        'rightIndex': rightIndex}

        # Initialize arrays/dictionary to store peak measurements
        self.ind_peak_widths = np.zeros(shape=(self.num_channels, self.total_bins))
        self.ind_peak_maxs = np.zeros(shape=(self.num_channels, self.total_bins))
        self.ind_peak_mins = np.zeros(shape=(self.num_channels, self.total_bins))
        self.ind_peak_props = {}

        # Loop through channels and bins for standard or kymograph analysis
        if self.analysis_type != "rolling":
            for channel in range(self.num_channels):
                for bin in range(self.total_bins):
                    if self.analysis_type == "standard":
                        signal = sig.savgol_filter(self.bin_values[:,channel, bin], window_length = 11, polyorder = 2)   
                    else:                     
                        signal = sig.savgol_filter(self.bin_values[channel, bin], window_length = 11, polyorder = 2)   
                    indv_props(channel, signal, bin)

        # If rolling analysis
        elif self.analysis_type == "rolling":
            self.ind_peak_widths = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.ind_peak_maxs = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.ind_peak_mins = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))

            its = self.num_submovies*self.num_channels*self.xpix*self.ypix
            with tqdm(total = its, miniters=its/100) as pbar:
                pbar.set_description('Peak Props: ')
                for submovie in range(self.num_submovies):
                    for channel in range(self.num_channels):
                        for bin in range(self.total_bins):
                            pbar.update(1)
                            signal = sig.savgol_filter(self.bin_values[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, channel, bin], window_length=11, polyorder=2)
                            indv_props(channel, signal, bin, submovie = submovie)

        # Calculate additional peak properties
        self.ind_peak_amps = self.ind_peak_maxs - self.ind_peak_mins
        self.ind_peak_rel_amps = self.ind_peak_amps / self.ind_peak_mins

      
        return self.ind_peak_widths, self.ind_peak_maxs, self.ind_peak_mins, self.ind_peak_amps, self.ind_peak_rel_amps, self.ind_peak_props

    def calc_indv_ACFs(self, peak_thresh=0.1):
        
        self.acfs, self.periods = calc_indv_ACFs_periods(num_channels=self.num_channels, 
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

        return self.acfs, self.periods

    def calc_indv_CCFs(self):
        
        self.indv_shifts, self.indv_ccfs, self.channel_combos = calc_indv_CCFs_shifts_channelCombos(num_channels=self.num_channels, 
                                                                                                    total_bins=self.total_bins,
                                                                                                    num_frames=self.num_frames, 
                                                                                                    bin_values=self.bin_values, 
                                                                                                    analysis_type=self.analysis_type, 
                                                                                                    roll_size=self.roll_size, 
                                                                                                    roll_by=self.roll_by, 
                                                                                                    num_submovies=self.num_submovies, 
                                                                                                    periods=self.periods
                                                                                                    )

        return self.indv_shifts, self.indv_ccfs, self.channel_combos

############################################
########### INDIVIDUAL BIN PLOTS ###########
############################################
    
    def plot_indv_peak_props(self):
        """
        This method generates and plots individual peak properties for each channel and bin.

        Returns:
            - dict: Dictionary containing generated figures of individual peak property plots.
        """
        def return_figure(bin_signal: np.ndarray, prop_dict: dict, Ch_name: str):
            '''
            Space saving function to generate the plots for the individual peak prop plots
            '''
            # Extract peak properties from the dictionary
            smoothed_signal = prop_dict['smoothed']
            peaks = prop_dict['peaks']
            proms = prop_dict['proms']
            heights = prop_dict['heights']
            leftIndex = prop_dict['leftIndex']
            rightIndex = prop_dict['rightIndex']

            # Create the figure and plot raw and smoothed signals
            fig, ax = plt.subplots()
            ax.plot(bin_signal, color = 'tab:gray', label = 'raw signal')
            ax.plot(smoothed_signal, color = 'tab:cyan', label = 'smoothed signal')

            # Plot each peak width and amplitude
            if not np.isnan(peaks).any():
                for i in range(peaks.shape[0]):
                    ax.hlines(heights[i], 
                            leftIndex[i], 
                            rightIndex[i], 
                            color='tab:olive', 
                            linestyle = '-')
                    ax.vlines(peaks[i], 
                            smoothed_signal[peaks[i]]-proms[i],
                            smoothed_signal[peaks[i]], 
                            color='tab:purple', 
                            linestyle = '-')
                # Plot the legend for the first peak
                ax.hlines(heights[0], 
                        leftIndex[0], 
                        rightIndex[0], 
                        color='tab:olive', 
                        linestyle = '-',
                        label='FWHM')
                ax.vlines(peaks[0], 
                        smoothed_signal[peaks[0]]-proms[0],
                        smoothed_signal[peaks[0]], 
                        color='tab:purple', 
                        linestyle = '-',
                        label = 'Peak amplitude')
                
                ax.legend(loc='upper right', fontsize='small', ncol=1)
                ax.set_xlabel('Time (frames)')
                ax.set_ylabel('Signal (AU)')
                ax.set_title(f'{Ch_name} peak properties')
            plt.close(fig)
            return fig

        # Dictionary to store generated figures
        self.indv_peak_figs = {}

        # Generate plots for each channel
        if hasattr(self, 'ind_peak_widths'):
            its = self.num_channels*self.total_bins
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind peaks')
                for channel in range(self.num_channels):
                    for bin in range(self.total_bins):
                        pbar.update(1)
                        to_plot = self.bin_values[:,channel, bin] if self.analysis_type == "standard" else self.bin_values[channel,bin, :]
                        # Generate and store the figure for the current channel and bin
                        self.indv_peak_figs[f'Ch{channel + 1} Bin {bin + 1} Peak Props'] = return_figure(to_plot,
                                                                                                    self.ind_peak_props[f'Ch {channel} Bin {bin}'],
                                                                                                    f'Ch{channel + 1} Bin {bin + 1}')

        return self.indv_peak_figs

    def plot_indv_acfs(self):
        """
        This method generates and plots individual autocorrelation functions (ACFs) for each channel and bin.

        Returns:
            - dict: Dictionary containing generated figures of individual ACF plots.
        """
        def return_figure(raw_signal: np.ndarray, acf_curve: np.ndarray, Ch_name: str, period: int):
            '''
            Space saving function to generate the plots for the individual ACF plots
            '''
            # Create subplots for raw signal and autocorrelation curve
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(raw_signal)
            ax1.set_xlabel(f'{Ch_name} Raw Signal')
            ax1.set_ylabel('Mean bin px value')
            ax2.plot(np.arange(-self.num_frames + 1, self.num_frames), acf_curve)
            ax2.set_ylabel('Autocorrelation')
            
            # Annotate the first peak identified as the period if available
            if not period == np.nan:
                color = 'red'
                ax2.axvline(x = period, alpha = 0.5, c = color, linestyle = '--')
                ax2.axvline(x = -period, alpha = 0.5, c = color, linestyle = '--')
                ax2.set_xlabel(f'Period is {period} frames')
            else:
                ax2.set_xlabel(f'No period identified')

            fig.subplots_adjust(hspace=0.5)
            plt.close(fig)
            return(fig)

        # Empty dictionary to store generated figures
        self.indv_acf_plots = {}

        # Iterate through channels and bins to plot individual autocorrelation curves
        its = self.num_channels*self.total_bins
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('ind acfs')
            for channel in range(self.num_channels):
                for bin in range(self.total_bins):
                    pbar.update(1) 
                    to_plot = self.bin_values[:,channel, bin] if self.analysis_type == "standard" else self.bin_values[channel,bin, :]
                    # Generate and store the figure for the current channel and bin
                    self.indv_acf_plots[f'Ch{channel + 1} Bin {bin + 1} ACF'] = return_figure(to_plot, 
                                                                                            self.acfs[channel, bin], 
                                                                                            f'Ch{channel + 1}', 
                                                                                            self.periods[channel, bin])
        return self.indv_acf_plots

    def plot_indv_ccfs(self, save_folder):
        """
        This method generates and plots individual cross-correlation functions (CCFs) for each channel and bin.

        It then saves the measurements to CSV files for each channel combination and bin.

        Parameters:
            - save_folder (str): Path to the folder where CSV files will be saved.

        Returns:
            - dict: Dictionary containing generated figures of individual CCF plots.
        """
        # Create subplots for raw signals and cross-correlation curve
        def return_figure(ch1: np.ndarray, ch2: np.ndarray, ccf_curve: np.ndarray, ch1_name: str, ch2_name: str, shift: int):
            '''
            Space saving function to generate the plots for the individual CCF plots
            '''
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(ch1, color = 'tab:blue', label = ch1_name)
            ax1.plot(ch2, color = 'tab:orange', label = ch2_name)
            ax1.set_xlabel('time (frames)')
            ax1.set_ylabel('Mean bin px value')
            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax2.plot(np.arange(-self.num_frames + 1, self.num_frames), ccf_curve)
            ax2.set_ylabel('Crosscorrelation')
            
            # Annotate the first peak identified as the shift if available
            if not shift == np.nan:
                color = 'red'
                ax2.axvline(x = shift, alpha = 0.5, c = color, linestyle = '--')
                if shift < 1:
                    ax2.set_xlabel(f'{ch1_name} leads by {int(abs(shift))} frames')
                elif shift > 1:
                    ax2.set_xlabel(f'{ch2_name} leads by {int(abs(shift))} frames')
                else:
                    ax2.set_xlabel('no shift detected')
            else:
                ax2.set_xlabel(f'No peaks identified')
            
            fig.subplots_adjust(hspace=0.5)
            plt.close(fig)
            return(fig)
        
        def normalize(signal: np.ndarray):
            # Normalize between 0 and 1
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        # Empty dictionary to store generated figures
        self.indv_ccf_plots = {}

        # Iterate through channel combinations and bins to plot individual cross-correlation curves
        if self.num_channels > 1:
            its = len(self.channel_combos)*self.total_bins
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind ccfs')
                for combo_number, combo in enumerate(self.channel_combos):
                    for bin in range(self.total_bins):
                        pbar.update(1)
                        to_plot1 = self.bin_values[:, combo[0], bin] if self.analysis_type == "standard" else self.bin_values[combo[0], bin, :]
                        to_plot2 = self.bin_values[:, combo[1], bin] if self.analysis_type == "standard" else self.bin_values[combo[1], bin, :]
                        # Generate and store the figure for the current channel combination and bin
                        self.indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = return_figure(ch1 = normalize(to_plot1),
                                                                                                        ch2 = normalize(to_plot2),
                                                                                                        ccf_curve = self.indv_ccfs[combo_number, bin],
                                                                                                        ch1_name = f'Ch{combo[0] + 1}',
                                                                                                        ch2_name = f'Ch{combo[1] + 1}',
                                                                                                        shift = self.indv_shifts[combo_number, bin])
                        
                        # Save the individual bin values
                        ccf_curve = self.indv_ccfs[combo_number, bin]
                        measurements = list(zip_longest(range(1, len(ccf_curve) + 1),  normalize(to_plot1), normalize(to_plot2), ccf_curve, fillvalue=None))
                        indv_ccfs_filename = os.path.join(save_folder, f'Bin {bin + 1}_CCF_values.csv')
                    
                        # Write measurements to CSV file
                        with open(indv_ccfs_filename, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(['Time', 'Ch1_Value', 'Ch2_Value', 'CCF_Value'])
                            for time, ch1_val, ch2_val, ccf_val in measurements:
                                writer.writerow([time, ch1_val, ch2_val, ccf_val])
        
        return self.indv_ccf_plots

############################################
############## MEAN BIN PLOTS ##############
############################################

    def plot_mean_peak_props(self):
        """
        This method generates and plots histograms and boxplots for the mean peak properties
        (minimum value, maximum value, amplitude, and width) for each channel.

        Returns:
            - dict: A dictionary containing generated figures of mean peak property plots for each channel.
        """
        def return_figure(min_array: np.ndarray, max_array: np.ndarray, amp_array: np.ndarray, width_array: np.ndarray, Ch_name: str):
            '''
            Space saving function to generate the plots for the mean peak prop plots
            '''
            # Create subplots for histograms and boxplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            # Filter out NaN values from arrays
            min_array = [val for val in min_array if not np.isnan(val)]
            max_array = [val for val in max_array if not np.isnan(val)]
            amp_array = [val for val in amp_array if not np.isnan(val)]
            width_array = [val for val in width_array if not np.isnan(val)]

            # Define plot parameters for histograms and boxplots
            plot_params = { 'amp' : (amp_array, 'tab:blue'),
                            'min' : (min_array, 'tab:purple'),
                            'max' : (max_array, 'tab:orange')}
            
            # Plot histograms for peak properties
            for labels, (arr, arr_color) in plot_params.items():
                ax1.hist(arr, color = arr_color, label = labels, alpha = 0.75)

            # Plot boxplots for peak properties
            boxes = ax2.boxplot([val[0] for val in plot_params.values()], patch_artist = True)
            ax2.set_xticklabels(plot_params.keys())
            for box, box_color in zip(boxes['boxes'], [val[1] for val in plot_params.values()]):
                box.set_color(box_color)

            # Set labels and legends for histograms and boxplots
            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax1.set_xlabel(f'{Ch_name} histogram of peak values')
            ax1.set_ylabel('Occurances')
            ax2.set_xlabel(f'{Ch_name} boxplot of peak values')
            ax2.set_ylabel('Value (AU)')
            
            # Plot histogram for peak widths
            ax3.hist(width_array, color = 'dimgray', alpha = 0.75)
            ax3.set_xlabel(f'{Ch_name} histogram of peak widths')
            ax3.set_ylabel('Occurances')

            # Plot boxplot for peak widths
            bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('dimgray')
            ax4.set_xlabel(f'{Ch_name} boxplot of peak widths')
            ax4.set_ylabel('Peak width (frames)')

            fig.subplots_adjust(hspace=0.6, wspace=0.6)
            plt.close(fig)
            return fig

        # Empty dictionary to fill with figures for each channel
        self.peak_figs = {}
    
        if hasattr(self, 'ind_peak_widths'):
            for channel in range(self.num_channels):
                self.peak_figs[f'Ch{channel + 1} Peak Props'] = return_figure(self.ind_peak_mins[channel], 
                                                                              self.ind_peak_maxs[channel], 
                                                                              self.ind_peak_amps[channel], 
                                                                              self.ind_peak_widths[channel], 
                                                                              f'Ch{channel + 1}')

        return self.peak_figs
    
    def plot_mean_ACF(self):
        """
        This method generates and plots the mean autocorrelation curve with shaded standard deviation area,
        a histogram of period values, and a boxplot of period values for each channel.

        Returns:
            - dict: Dictionary containing generated figures of mean ACF plots.
        """
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel: str):
            '''
            Space saving function to generate the plots for the mean ACF plots
            '''
            # Plot mean autocorrelation curve with shaded area representing standard deviation
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_frames + 1, self.num_frames)

            # Create the figure with subplots
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            
            # Plot mean autocorrelation curve with shaded area representing standard deviation
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel} Mean Autocorrelation Curve ± Standard Deviation') 

            # Plot histogram of period values
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of period values (frames)')
            ax['B'].set_ylabel('Occurances')

            # Plot boxplot of period values
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of period values')
            ax['C'].set_ylabel(f'Measured period (frames)')

            fig.subplots_adjust(hspace=0.25, wspace=0.5)  
            plt.close(fig)
            return fig

        # Dictionary to store generated figures
        self.acf_figs = {}
        
        if hasattr(self, 'acfs'):
            # Generate plots for each channel
            for channel in range(self.num_channels):
                self.acf_figs[f'Ch{channel + 1} Mean ACF'] = return_figure(self.acfs[channel], 
                                                                         self.periods[channel], 
                                                                         f'Ch{channel + 1}')        

        return self.acf_figs
    
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
        
        if hasattr(self, 'periods'):
            for channel in range(self.num_channels):
                self.plot_list[f'Ch {channel + 1} Period'] = return_plot('Submovie',
                                                                          f'Ch {channel + 1} Mean Period',
                                                                          f'Ch {channel + 1} StdDev Period',
                                                                          f'Ch {channel + 1} Mean ± StdDev Period (frames)')
        
        if hasattr(self, 'shifts'):
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

            if hasattr(self, 'acfs'):
                submovie_periods_with_stats = add_stats(self.periods[submovie], 'Period')
                for channel in range(self.num_channels):
                    statified_measurements.append(submovie_periods_with_stats[channel])
            
            if hasattr(self, 'ccfs'):
                submovie_shifts_with_stats = add_stats(self.ccfs[submovie], 'Shift')
                for combo_number, _ in enumerate(self.channel_combos):
                    statified_measurements.append(submovie_shifts_with_stats[combo_number])
            
            if hasattr(self, 'peak_widths'):
                submovie_widths_with_stats = add_stats(self.ind_peak_widths[submovie], 'Peak Width')
                submovie_maxs_with_stats = add_stats(self.ind_peak_maxs[submovie], 'Peak Max')
                submovie_mins_with_stats = add_stats(self.ind_peak_mins[submovie], 'Peak Min')
                submovie_amps_with_stats = add_stats(self.ind_peak_amps[submovie], 'Peak Amp')
                submovie_rel_amps_with_stats = add_stats(self.ind_peak_rel_amps[submovie], 'Peak Rel Amp')
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
            
            if hasattr(self, 'acfs'):
                for channel in range(self.num_channels):
                    pcnt_no_period = (np.count_nonzero(np.isnan(self.periods[submovie, channel])) / self.total_bins) * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(self.periods[submovie, channel])

            if hasattr(self, 'indv_ccfs'):
                for combo_number, combo in enumerate(self.channel_combos):
                    pcnt_no_shift = np.count_nonzero(np.isnan(self.indv_ccfs[submovie, combo_number])) / self.total_bins * 100
                    submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(self.indv_shifts[submovie, combo_number])

            if hasattr(self, 'ind_peak_widths'):
                for channel in range(self.num_channels):
                    # using widths, but because these are all assigned together it applies to all peak properties
                    pcnt_no_peaks = np.count_nonzero(np.isnan(self.ind_peak_widths[submovie, channel])) / self.total_bins * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Width'] = func(self.ind_peak_widths[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Max'] = func(self.ind_peak_maxs[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Min'] = func(self.ind_peak_mins[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Amp'] = func(self.ind_peak_amps[submovie, channel])
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
            pcnt_no_period = [np.count_nonzero(np.isnan(self.periods[channel])) / self.periods[channel].shape[0] * 100 for channel in range(self.num_channels)]
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
            pcnt_no_peaks = [np.count_nonzero(np.isnan(self.ind_peak_widths[channel])) / self.ind_peak_widths[channel].shape[0] * 100 for channel in range(self.num_channels)]
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
        if hasattr(self, 'acfs'):
            self.periods_with_stats = add_stats(self.periods, 'Period')
            for channel in range(self.num_channels):
                statified_measurements.append(self.periods_with_stats[channel])

        if hasattr(self, 'indv_ccfs'):
            self.shifts_with_stats = add_stats(self.indv_shifts, 'Shift')
            for combo_number, combo in enumerate(self.channel_combos):
                statified_measurements.append(self.shifts_with_stats[combo_number])

        if hasattr(self, 'ind_peak_widths'):
            self.peak_widths_with_stats = add_stats(self.ind_peak_widths, 'Peak Width')
            self.peak_maxs_with_stats = add_stats(self.ind_peak_maxs, 'Peak Max')
            self.peak_mins_with_stats = add_stats(self.ind_peak_mins, 'Peak Min')
            self.peak_amps_with_stats = add_stats(self.ind_peak_amps, 'Peak Amp')
            self.peak_relamp_with_stats = add_stats(self.ind_peak_rel_amps, 'Peak Rel Amp')
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