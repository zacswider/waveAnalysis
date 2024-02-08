import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal as sig
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from itertools import zip_longest
from tifffile import imread, TiffFile

np.seterr(divide='ignore', invalid='ignore')

class TotalSignalProcessor:
    def __init__(self, analysis_type, image_path, kern=None, step=None, roll_size=None, roll_by=None, line_width=None):
        # Import variables
        self.roll_size = roll_size
        self.roll_by = roll_by
        self.line_width = line_width
        self.kernel_size = kern
        self.step = step
        self.analysis_type = analysis_type

        # Image import
        self.image_path = image_path
        self.image = imread(self.image_path)
        with TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)
        self.standardize_image_dimensions(metadata)

        # Specific functions for rolling analysis
        self.check_and_set_rolling_parameters()

        # calculate the bin (box or line) values for each movie
        self.bin_values = self.calculate_bin_values()

    def standardize_image_dimensions(self, metadata):
        '''
        Extract metadata, and reshape the image
        '''
        if self.analysis_type != "kymograph":
            self.num_frames = metadata.get('frames', 1)
            self.num_slices = metadata.get('slices', 1)
            self.image = self.image.reshape(self.num_frames, self.num_slices, self.num_channels, *self.image.shape[-2:])

            # Max project if multiple slices
            if self.num_slices > 1:
                print('Max projecting image stack')
                self.image = np.max(self.image, axis=1)
                self.num_slices = 1
                self.image = self.image.reshape(self.num_frames, self.num_slices, self.num_channels, *self.image.shape[-2:])
        else:
            # we are either binning the image into boxes (standard) or columns (kymographs), so just call bins for simplicity
            self.total_bins = self.image.shape[-1] 
            # the number of rows in a kymograph is equal to the number to number of frames, so just call frames for simplicity
            self.num_frames = self.image.shape[-2] 
            self.image = self.image.reshape(self.num_frames, self.num_channels, self.total_bins)
            print(self.image.shape)

    def check_and_set_rolling_parameters(self):
        '''
        Specific parameters that are only set in the rolling analysis
        '''
        if self.analysis_type == "rolling":
            assert isinstance(self.roll_size, int) and isinstance(self.roll_by, int), 'Roll size and roll by must be integers'
            self.num_submovies = (self.num_frames - self.roll_size) // self.roll_by

    def calculate_bin_values(self):
        '''
        Calculate the mean signal for the specified box or line size over the images.
        '''
        # Use boxes for the standard and rolling analysis
        if self.analysis_type != "kymograph":
            # Calculate the index for the center of the kernel
            ind = self.kernel_size // 2
            # Apply uniform filter to calculate mean signal over specified box size
            box_values = nd.uniform_filter(self.image[:, 0, :, :, :], size=(1, 1, self.kernel_size, self.kernel_size))[:, :, ind::self.step, ind::self.step]
            # Get the dimensions of the resulting mean image
            self.xpix, self.ypix = box_values.shape[-2:]
            # We are either binning the image into boxes (standard) or columns (kymographs), so just call bins for simplicity
            self.total_bins = self.xpix * self.ypix
            box_values = box_values.reshape(self.num_frames, self.num_channels, self.total_bins)

            return box_values

        # Use lines for kymograph analysis
        else:
            line_values = np.zeros(shape=(self.num_frames, self.num_channels, self.total_bins))

            for channel in range(self.num_channels):
                for frame_num in range(self.num_frames):
                    for line_num in range(self.total_bins):
                        if self.line_width == 1:
                            pixel = self.image[frame_num, channel, line_num]
                            line_values[frame_num, channel, line_num] = pixel
                        elif self.line_width % 2 != 0:
                            line_width_extra = (self.line_width - 1) // 2
                            left_bound = max(0, line_num - line_width_extra)
                            right_bound = min(self.total_bins, line_num + line_width_extra + 1)
                            line_slice = self.image[frame_num, channel, left_bound:right_bound]
                            pixel = np.mean(line_slice)
                            line_values[frame_num, channel, line_num] = pixel
                        else:
                            raise ValueError("Line width must be odd!")

            return line_values

############################################
######## INDIVIDUAL BIN CALCULATION ########
############################################

    def calc_indv_ACFs(self, peak_thresh=0.1):
        """
        This method computes the autocorrelation functions (ACFs) for each channel and bin of the analyzed data.
        It also identifies peaks in the ACF curves to estimate periods.

        Parameters:
            - peak_thresh (float): Threshold for peak detection in the ACF curves. Defaults to 0.1.

        Returns:
            - acfs (numpy.ndarray): Array of autocorrelation functions.
            - periods (numpy.ndarray): Array of periods estimated from the ACF peaks.
        """
        def norm_and_calc_shifts(signal, num_frames_or_rollsize):
            """
            This function normalizes the input signal and computes the autocorrelation curve.
            It identifies peaks in the autocorrelation curve to estimate the delay.

            Parameters:
                - signal (numpy.ndarray): Input signal.
                - num_frames_or_rows_or_rollsize (int): Number of frames or roll size for normalization.

            Returns:
                - delay (float): Delay estimated from the autocorrelation curve.
                - acf_curve (numpy.ndarray): Autocorrelation curve of the normalized signal.
            """
            corr_signal = signal - np.mean(signal)
            acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
            # Normalize the autocorrelation curve
            acf_curve = acf_curve / (num_frames_or_rollsize * np.std(signal) ** 2)
            # Find peaks in the autocorrelation curve
            peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
            # Calculate absolute differences between peaks and center
            peaks_abs = np.abs(peaks - acf_curve.shape[0] // 2)
            # If peaks are identified, pick the closest one to the center
            if len(peaks) > 1:
                delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
            else:
                # Otherwise, return NaNs for both delay and autocorrelation curve
                delay = np.nan
                acf_curve = np.full((num_frames_or_rollsize * 2 - 1), np.nan)
            return delay, acf_curve
        
        # Initialize arrays to store period measurements and autocorrelation curves
        self.periods = np.zeros(shape=(self.num_channels, self.total_bins))
        self.acfs = np.zeros(shape=(self.num_channels, self.total_bins, self.num_frames * 2 - 1))

        # Loop through channels and bins for standard or kymograph analysis
        if self.analysis_type != "rolling":
            for channel in range(self.num_channels):
                for bin in range(self.total_bins):
                    signal = self.bin_values[:, channel, bin]
                    delay, acf_curve = norm_and_calc_shifts(signal, num_frames_or_rollsize=self.num_frames)
                    self.periods[channel, bin] = delay
                    self.acfs[channel, bin] = acf_curve
        # If rolling analysis
        elif self.analysis_type == "rolling":
            self.periods = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.acfs = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins, self.roll_size * 2 - 1))
            # Loop through submovies, channels, and bins
            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for bin in range(self.total_bins):
                        # Extract signal for rolling autocorrelation calculation
                        signal = self.bin_values[self.roll_by * submovie: self.roll_size + self.roll_by * submovie, channel, bin]
                        delay, acf_curve = norm_and_calc_shifts(signal, num_frames_or_rollsize=self.roll_size)
                        self.periods[submovie, channel, bin] = delay
                        self.acfs[submovie, channel, bin] = acf_curve
        return self.acfs, self.periods

    def calc_indv_CCFs(self):
        """
        This method computes the cross-correlation functions (CCFs) for each combination of channels.
        It also identifies peaks in the CCF curves to estimate shifts.

        Returns:
            - indv_shifts (numpy.ndarray): Array of shifts between signals.
            - indv_ccfs (numpy.ndarray): Array of cross-correlation functions.
            - channel_combos (list): List of channel combinations.
        """
        def calc_shifts(signal1, signal2, prominence=0.1, rolling = False):
            """
            This function calculates the shifts and cross-correlation curves between two signals.
            It performs signal smoothing, peak finding, and computes the cross-correlation curve.

            Parameters:
                - signal1 (numpy.ndarray): First input signal.
                - signal2 (numpy.ndarray): Second input signal.
                - prominence (float): Minimum prominence of peaks for peak finding. Defaults to 0.1.
                - rolling (bool): Flag indicating if the analysis is rolling. Defaults to False.

            Returns:
                - delay_frames (float): Delay between the signals.
                - cc_curve (numpy.ndarray): Cross-correlation curve of the signals.
            """
            # Smoothing signals and finding peaks
            signal1 = sig.savgol_filter(signal1, window_length=11, polyorder=3)
            signal2 = sig.savgol_filter(signal2, window_length=11, polyorder=3)
            peaks1, _ = sig.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
            peaks2, _ = sig.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

            # If peaks are found in both signals
            if len(peaks1) > 0 and len(peaks2) > 0:
                corr_signal1 = signal1 - signal1.mean()
                corr_signal2 = signal2 - signal2.mean()
                # Calculate cross-correlation curve
                cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
                if rolling:
                    cc_curve = cc_curve / (self.roll_size * signal1.std() * signal2.std())
                else:
                    cc_curve = sig.savgol_filter(cc_curve, window_length=11, polyorder=3)
                    cc_curve = cc_curve / (self.num_frames * signal1.std() * signal2.std())
                # Find peaks in the cross-correlation curve
                peaks, _ = sig.find_peaks(cc_curve, prominence=prominence)
                peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
                # If multiple peaks found, select the one closest to the center
                if len(peaks) > 1:
                    delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                    delayIndex = peaks[delay]
                    delay_frames = delayIndex - cc_curve.shape[0] // 2
                # Otherwise, return NaNs
                else:
                    delay_frames = np.nan
                    cc_curve = np.full((self.roll_size*2-1 if rolling else self.num_frames * 2 - 1), np.nan)
            else:
                # If no peaks found, return NaNs
                delay_frames = np.nan
                cc_curve = np.full((self.roll_size*2-1 if rolling else self.num_frames * 2 - 1), np.nan)

            return delay_frames, cc_curve
        
        # Initialize arrays to store shifts and cross-correlation curves
        channels = list(range(self.num_channels))
        self.channel_combos = []
        for i in range(self.num_channels):
            for j in channels[i+1:]:
                self.channel_combos.append([channels[i],j])
        num_combos = len(self.channel_combos)

        # Initialize arrays to store shifts and cross-correlation curves
        self.indv_shifts = np.zeros(shape=(num_combos, self.total_bins))
        self.indv_ccfs = np.zeros(shape=(num_combos, self.total_bins, self.num_frames*2-1))

        # Loop through combos for standard or kymograph analysis
        if self.analysis_type != "rolling":
            for combo_number, combo in enumerate(self.channel_combos):
                for bin in range(self.total_bins):
                    signal1 = self.bin_values[:, combo[0], bin]
                    signal2 = self.bin_values[:, combo[1], bin]
     
                    delay_frames, cc_curve = calc_shifts(signal1, signal2, prominence=0.1)

                    # The script has issues when the shift is very small or none, so minus the average period from the two channels
                    average_period = np.mean(self.periods[:, bin])
                    if abs(delay_frames) > abs(average_period * .6):
                        if delay_frames < 0:
                            delay_frames = delay_frames + average_period
                        elif delay_frames > 0:
                            delay_frames = delay_frames - average_period

                    self.indv_shifts[combo_number, bin] = delay_frames
                    self.indv_ccfs[combo_number, bin] = cc_curve

        # If rolling analysis
        elif self.analysis_type == "rolling":
            # Initialize arrays to store shifts and cross-correlation curves
            self.indv_shifts = np.zeros(shape=(self.num_submovies, num_combos, self.total_bins))
            self.indv_ccfs = np.zeros(shape=(self.num_submovies, num_combos, self.total_bins, self.roll_size*2-1))

            for submovie in range(self.num_submovies):
                for combo_number, combo in enumerate(self.channel_combos):
                    for bin in range(self.total_bins):
                        signal1 = self.bin_values[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[0], bin]
                        signal2 = self.bin_values[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[1], bin]

                        delay_frames, cc_curve = calc_shifts(signal1, signal2, prominence=0.1, rolling = True)

                        self.indv_shifts[submovie, combo_number, bin] = delay_frames
                        self.indv_ccfs[submovie, combo_number, bin] = cc_curve

        return self.indv_shifts, self.indv_ccfs, self.channel_combos

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
        def indv_props(signal, bin, submovie = None):
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
                    signal = sig.savgol_filter(self.bin_values[:,channel, bin], window_length = 11, polyorder = 2)                       
                    indv_props(signal, bin)

        # If rolling analysis
        elif self.analysis_type == "rolling":
            self.ind_peak_widths = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.ind_peak_maxs = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.ind_peak_mins = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))

            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for bin in range(self.total_bins):
                        signal = sig.savgol_filter(self.bin_values[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, channel, bin], window_length=11, polyorder=2)
                        indv_props(signal, bin, submovie = submovie)

        # Calculate additional peak properties
        self.ind_peak_amps = self.ind_peak_maxs - self.ind_peak_mins
        self.ind_peak_rel_amps = self.ind_peak_amps / self.ind_peak_mins

      
        return self.ind_peak_widths, self.ind_peak_maxs, self.ind_peak_mins, self.ind_peak_amps, self.ind_peak_rel_amps, self.ind_peak_props

############################################
########### INDIVIDUAL BIN PLOTS ###########
############################################
    
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
                    # Generate and store the figure for the current channel and bin
                    self.indv_acf_plots[f'Ch{channel + 1} Bin {bin + 1} ACF'] = return_figure(self.bin_values[:,channel, bin], 
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
                        # Generate and store the figure for the current channel combination and bin
                        self.indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = return_figure(ch1 = normalize(self.bin_values[:, combo[0], bin]),
                                                                                                        ch2 = normalize(self.bin_values[:, combo[1], bin]),
                                                                                                        ccf_curve = self.indv_ccfs[combo_number, bin],
                                                                                                        ch1_name = f'Ch{combo[0] + 1}',
                                                                                                        ch2_name = f'Ch{combo[1] + 1}',
                                                                                                        shift = self.indv_shifts[combo_number, bin])
                        
                        # Save the individual bin values
                        ccf_curve = self.indv_ccfs[combo_number, bin]
                        measurements = list(zip_longest(range(1, len(ccf_curve) + 1),  normalize(self.bin_values[:, combo[0], bin]), normalize(self.bin_values[:, combo[1], bin]), ccf_curve, fillvalue=None))
                        indv_ccfs_filename = os.path.join(save_folder, f'Bin {bin + 1}_CCF_values.csv')
                    
                        # Write measurements to CSV file
                        with open(indv_ccfs_filename, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(['Time', 'Ch1_Value', 'Ch2_Value', 'CCF_Value'])
                            for time, ch1_val, ch2_val, ccf_val in measurements:
                                writer.writerow([time, ch1_val, ch2_val, ccf_val])
        
        return self.indv_ccf_plots

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
                        # Generate and store the figure for the current channel and bin
                        self.indv_peak_figs[f'Ch{channel + 1} Bin {bin + 1} Peak Props'] = return_figure(self.bin_values[:,channel, bin],
                                                                                                    self.ind_peak_props[f'Ch {channel} Bin {bin}'],
                                                                                                    f'Ch{channel + 1} Bin {bin + 1}')

        return self.indv_peak_figs

############################################
############## MEAN BIN PLOTS ##############
############################################

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
            
            if hasattr(self, 'periods'):
                for channel in range(self.num_channels):
                    pcnt_no_period = (np.count_nonzero(np.isnan(self.periods[submovie, channel])) / self.total_bins) * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(self.periods[submovie, channel])

            if hasattr(self, 'shifts'):
                for combo_number, combo in enumerate(self.channel_combos):
                    pcnt_no_shift = np.count_nonzero(np.isnan(self.indv_ccfs[submovie, combo_number])) / self.total_bins * 100
                    submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(self.indv_shifts[submovie, combo_number])

            if hasattr(self, 'peak_widths'):
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
        
        
        if self.analysis_type != "rolling":
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

        else: 

            self.submovie_measurements = []

            for submovie in range(self.num_submovies):
                statified_measurements = []

                if hasattr(self, 'acfs'):
                    submovie_periods_with_stats = add_stats(self.periods[submovie], 'Period')
                    for channel in range(self.num_channels):
                        statified_measurements.append(submovie_periods_with_stats[channel])
                
                if hasattr(self, 'indv_ccfs'):
                    submovie_shifts_with_stats = add_stats(self.indv_ccfs[submovie], 'Shift')
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