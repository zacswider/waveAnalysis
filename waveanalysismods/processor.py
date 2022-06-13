import numpy as np
import pandas as pd
import scipy.signal as sig
from tqdm import tqdm
import matplotlib.pyplot as plt
from tifffile import imread, imwrite, TiffFile
import scipy.ndimage as nd
np.seterr(divide='ignore', invalid='ignore')

class TotalSignalProcessor:
    
    def __init__(self, image_path, kern, step):
        self.image_path = image_path
        self.kernel_size = kern
        self.image = imread(self.image_path)
        self.step  = step

        # standardize image dimensions
        with TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)
        self.num_slices = metadata.get('slices', 1)
        self.num_frames = metadata.get('frames', 1)
        self.image = self.image.reshape(self.num_frames, 
                                        self.num_slices, 
                                        self.num_channels, 
                                        self.image.shape[-2], 
                                        self.image.shape[-1])

        # max project image stack if num_slices > 1
        if self.num_slices > 1:
            print(f'Max projecting image stack')
            self.image = np.max(self.image, axis = 1)
            self.num_slices = 1
            self.image = self.image.reshape(self.num_frames, 
                                            self.num_slices, 
                                            self.num_channels, 
                                            self.image.shape[-2], 
                                            self.image.shape[-1])

        # return the time-axis means for each channel
        ind = kern // 2
        self.means = nd.uniform_filter(self.image[:,0,:,:,:], size = (1,1,kern,kern))[:,:,ind:-ind:step, ind:-ind:step]
        self.xpix = self.means.shape[-2]
        self.ypix = self.means.shape[-1]
        self.num_boxes = self.xpix*self.ypix
        self.means = self.means.reshape(self.means.shape[0], self.means.shape[1], self.num_boxes)

    # function to return the autocorrelation of each box in the image stack for each channel
    def calc_ACF(self, peak_thresh=0.1):
        '''

        '''
        # make empty arrays to populate with 1) period measurements and 2) acf curves
        self.periods = np.zeros(shape=(self.num_channels, self.num_boxes))
        self.acfs = np.zeros(shape=(self.num_channels, self.num_boxes, self.num_frames*2-1))

        for channel in range(self.num_channels):
            for box in range(self.num_boxes):
                # calculate full autocorrelation
                signal = self.means[:,channel, box]
                corr_signal = signal - signal.mean()
                acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
                # normalize the curve
                acf_curve = acf_curve / (self.num_frames * signal.std() ** 2)
                peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
                # absolute difference between each peak and zero
                peaks_abs = abs(peaks - acf_curve.shape[0]//2)
                # if peaks were identified, pick the one closest to the center
                if len(peaks) > 1:
                    delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
                # otherwise, return nans for both period and autocorrelation curve
                else:
                    delay = np.nan
                    acf_curve = np.full((self.num_frames*2-1), np.nan)
                self.periods[channel, box] = delay
                self.acfs[channel, box] = acf_curve

        return self.periods, self.acfs

    def calc_CCF(self):
        '''
        
        '''
        # make a list of unique channel combinations to calculate CCF for
        channels = list(range(self.num_channels))
        self.channel_combos = []
        for i in range(self.num_channels):
            for j in channels[i+1:]:
                self.channel_combos.append([channels[i],j])
        num_combos = len(self.channel_combos)

        # make empty arrays to populate with 1) period measurements and 2) acf curves   
        self.shifts = np.zeros(shape=(num_combos, self.num_boxes))
        self.ccfs = np.zeros(shape=(num_combos, self.num_boxes, self.num_frames*2-1))

        for combo_number, combo in enumerate(self.channel_combos):
            for box in range(self.num_boxes):
                # calculate full cross-correlation
                signal1 = self.means[:,combo[0], box]
                signal2 = self.means[:,combo[1], box]
                corr_signal1 = signal1 - signal1.mean()
                corr_signal2 = signal2 - signal2.mean()
                cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
                # normalize the curve
                cc_curve = cc_curve / (self.num_frames * signal1.std() * signal2.std())
                peaks, _ = sig.find_peaks(cc_curve, prominence=0.1)
                # absolute difference between each peak and zero
                peaks_abs = abs(peaks - cc_curve.shape[0]//2)
                # if peaks were identified, pick the one closest to the center
                if len(peaks) > 1:
                    delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
                # otherwise, return nans for both period and autocorrelation curve
                else:
                    delay = np.nan
                    cc_curve = np.full((self.num_frames*2-1), np.nan)
                self.shifts[combo_number, box] = delay
                self.ccfs[combo_number, box] = cc_curve

        return self.shifts, self.ccfs

    def calc_peak_props(self):
        '''
        
        '''
        # make empty arrays to fill with peak measurements for each channel
        self.peak_widths = np.zeros(shape=(self.num_channels, self.num_boxes))
        self.peak_maxs = np.zeros(shape=(self.num_channels, self.num_boxes))
        self.peak_mins = np.zeros(shape=(self.num_channels, self.num_boxes))
        
        for channel in range(self.num_channels):
            for box_num in range(self.num_boxes):

                signal = sig.savgol_filter(self.means[:,channel, box_num], window_length = 11, polyorder = 2)
                peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

                # if peaks detected, calculate properties and return property averages. Otherwise return nans
                if len(peaks) > 0:
                    proms, _, _ = sig.peak_prominences(signal, peaks)
                    widths, _, _, _ = sig.peak_widths(signal, peaks, rel_height=0.5)
                    mean_width = np.mean(widths, axis=0)
                    mean_max = np.mean(signal[peaks], axis = 0)
                    mean_min = np.mean(signal[peaks]-proms, axis = 0)
                    self.peak_widths[channel, box_num] = mean_width
                    self.peak_maxs[channel, box_num] = mean_max
                    self.peak_mins[channel, box_num] = mean_min
                else:
                    self.peak_widths[channel, box_num] = np.nan
                    self.peak_maxs[channel, box_num] = np.nan
                    self.peak_mins[channel, box_num] = np.nan

        self.peak_amps = self.peak_maxs - self.peak_mins
        self.peak_rel_amps = self.peak_amps / self.peak_mins

        return self.peak_widths, self.peak_maxs, self.peak_mins, self.peak_amps, self.peak_rel_amps

    # function to plot a summary of the period measurements
    def plot_mean_CF(self):
        '''

        '''
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel: str, type_of_plot: str, type_of_measurement: str):
            '''
            Space saving function for plotting the mean autocorrelation or crosscorrelation curve. Returns a figure object.
            '''
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_frames + 1, self.num_frames)
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel} Mean {type_of_plot} Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of {type_of_measurement} values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of {type_of_measurement} values')
            ax['C'].set_ylabel(f'Measured {type_of_measurement} (frames)')
            fig.subplots_adjust(hspace=0.25, wspace=0.5)   
            plt.close(fig)
            return fig

        # empty dict to fill with figures, in the event that we make more than one
        self.cf_figs = {}
        
        if hasattr(self, 'acfs'):
            # make a separate plot for each channel
            for channel in range(self.num_channels):
                self.cf_figs[f'Ch{channel + 1} Mean ACF'] = return_figure(self.acfs[channel], 
                                                                         self.periods[channel], 
                                                                         f'Ch{channel + 1}', 
                                                                         'Autocorrelation', 
                                                                         'period')        
        if hasattr(self, 'ccfs'):
            if self.num_channels > 1:
                for combo_number, combo in enumerate(self.channel_combos):
                    self.cf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_figure(self.ccfs[combo_number], 
                                                                                                self.shifts[combo_number], 
                                                                                                f'Ch{combo[0] + 1}-Ch{combo[1] + 1}', 
                                                                                                'Crosscorrelation', 
                                                                                                'shift')

        return self.cf_figs


    def plot_peak_props(self):
        '''
        
        '''
        def return_figure(min_array: np.ndarray, max_array: np.ndarray, amp_array: np.ndarray, width_array: np.ndarray, Ch_name: str):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # filter nans out of arrays
            min_array = [val for val in min_array if not np.isnan(val)]
            max_array = [val for val in max_array if not np.isnan(val)]
            amp_array = [val for val in amp_array if not np.isnan(val)]
            width_array = [val for val in width_array if not np.isnan(val)]

            plot_params = { 'amp' : (amp_array, 'tab:blue'),
                            'min' : (min_array, 'tab:purple'),
                            'max' : (max_array, 'tab:orange')}
            for labels, (arr, arr_color) in plot_params.items():
                ax1.hist(arr, color = arr_color, label = labels, alpha = 0.75)
            boxes = ax2.boxplot([val[0] for val in plot_params.values()], patch_artist = True)
            ax2.set_xticklabels(plot_params.keys())
            for box, box_color in zip(boxes['boxes'], [val[1] for val in plot_params.values()]):
                box.set_color(box_color)

            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax1.set_xlabel(f'{Ch_name} histogram of peak values')
            ax1.set_ylabel('Occurances')

            ax2.set_xlabel(f'{Ch_name} boxplot of peak values')
            ax2.set_ylabel('Value (AU)')
            
            ax3.hist(width_array, color = 'dimgray', alpha = 0.75)
            ax3.set_xlabel(f'{Ch_name} histogram of peak widths')
            ax3.set_ylabel('Occurances')
            bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('dimgray')
            ax4.set_xlabel(f'{Ch_name} boxplot of peak widths')
            ax4.set_ylabel('Peak width (frames)')
            fig.subplots_adjust(hspace=0.6, wspace=0.6)
            plt.close(fig)
            return fig

        # empty dictionary to fill with figures, in the event that we make more than one
        self.peak_figs = {}
        # fill the dictionary with plots for each channel
        if hasattr(self, 'peak_widths'):
            for channel in range(self.num_channels):
                self.peak_figs[f'Ch{channel + 1} Peak Props'] = return_figure(self.peak_mins[channel], 
                                                                              self.peak_maxs[channel], 
                                                                              self.peak_amps[channel], 
                                                                              self.peak_widths[channel], 
                                                                              f'Ch{channel + 1}')

        return self.peak_figs

    # function to summarize the results in the acf_results, ccf_results, and peak_results dictionaries as a dataframe
    def organize_measurements(self):
        '''

        '''
        
        # function to summarize measurments statistics by appending them to the beginning of the measurement list
        def add_stats(measurements: np.ndarray, measurement_name: str):
            '''
            Accepts a list of measurements. Calculates the mean, median, standard deviation, and SEM,
            and append them to the beginning of the list in that order. Finally, appends the name of
            the measurement of the beginning of the list.
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

        # column names for the dataframe summarizing the box results
        col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        col_names.extend([f'Box{i}' for i in range(self.num_boxes)])
        # combine all the statified measurements into a single list
        statified_measurements = []

        # insert Mean, Median, StdDev, and SEM into the beginning of each  list
        if hasattr(self, 'acfs'):
            self.periods_with_stats = add_stats(self.periods, 'Period')
            for channel in range(self.num_channels):
                statified_measurements.append(self.periods_with_stats[channel])

        if hasattr(self, 'ccfs'):
            self.shifts_with_stats = add_stats(self.shifts, 'Shift')
            for combo_number, combo in enumerate(self.channel_combos):
                statified_measurements.append(self.shifts_with_stats[combo_number])

        if hasattr(self, 'peak_widths'):
            self.peak_widths_with_stats = add_stats(self.peak_widths, 'Peak Width')
            self.peak_maxs_with_stats = add_stats(self.peak_maxs, 'Peak Max')
            self.peak_mins_with_stats = add_stats(self.peak_mins, 'Peak Min')
            self.peak_amps_with_stats = add_stats(self.peak_amps, 'Peak Amp')
            self.peak_relamp_with_stats = add_stats(self.peak_rel_amps, 'Peak Rel Amp')
            for channel in range(self.num_channels):
                statified_measurements.append(self.peak_widths_with_stats[channel])
                statified_measurements.append(self.peak_maxs_with_stats[channel])
                statified_measurements.append(self.peak_mins_with_stats[channel])
                statified_measurements.append(self.peak_amps_with_stats[channel])
                statified_measurements.append(self.peak_relamp_with_stats[channel])

        # and turn it into a dataframe
        self.im_measurements = pd.DataFrame(statified_measurements, columns = col_names)
        return self.im_measurements

    def summarize_image(self, file_name = None, group_name = None):
        '''

        '''
        # dictionary to store the summarized measurements for each image
        self.file_data_summary = {}
        
        if file_name:
            self.file_data_summary['File Name'] = file_name
        if group_name:
            self.file_data_summary['Group Name'] = group_name
        self.file_data_summary['Num Boxes'] = self.num_boxes

        stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

        if hasattr(self, 'periods_with_stats'):
            pcnt_no_period = [np.count_nonzero(np.isnan(self.periods[channel])) / self.periods[channel].shape[0] * 100 for channel in range(self.num_channels)]
            for channel in range(self.num_channels):
                self.file_data_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period[channel]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch {channel + 1} {stat} Period'] = self.periods_with_stats[channel][ind + 1]
        
        if hasattr(self, 'shifts_with_stats'):
            pcnt_no_shift = [np.count_nonzero(np.isnan(self.shifts[combo_number])) / self.shifts[combo_number].shape[0] * 100 for combo_number, combo in enumerate(self.channel_combos)]
            for combo_number, combo in enumerate(self.channel_combos):
                self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift[combo_number]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = self.shifts_with_stats[combo_number][ind + 1]

        if hasattr(self, 'peak_widths_with_stats'):
            # using widths, but because these are all assigned together it applies to all peak properties
            pcnt_no_peaks = [np.count_nonzero(np.isnan(self.peak_widths[channel])) / self.peak_widths[channel].shape[0] * 100 for channel in range(self.num_channels)]
            for channel in range(self.num_channels):
                self.file_data_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks[channel]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Width'] = self.peak_widths_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Max'] = self.peak_maxs_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Min'] = self.peak_mins_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Amp'] = self.peak_amps_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Rel Amp'] = self.peak_relamp_with_stats[channel][ind + 1]
            
        return self.file_data_summary


##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################

class RollingSignalProcessor: 
    
    def __init__(self, image_path, kern, step, roll_size = None, roll_by = None):
        self.image_path = image_path
        self.kernel_size = kern
        self.step  = step
        self.image = imread(self.image_path)
        self.roll_size = roll_size
        self.roll_by = roll_by

        # sanity checks
        assert type(self.roll_size) == int and type(self.roll_by) == int, 'Roll size and roll by must be integers'
        
        # standardize image dimensions
        with TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)
        self.num_slices = metadata.get('slices', 1)
        self.num_frames = metadata.get('frames', 1)
        self.image = self.image.reshape(self.num_frames, 
                                        self.num_slices, 
                                        self.num_channels, 
                                        self.image.shape[-2], 
                                        self.image.shape[-1])

        # max project image stack if num_slices > 1
        if self.num_slices > 1:
            print(f'Max projecting image stack')
            self.image = np.max(self.image, axis = 1)
            self.num_slices = 1
            self.image = self.image.reshape(self.num_frames, 
                                            self.num_slices, 
                                            self.num_channels, 
                                            self.image.shape[-2], 
                                            self.image.shape[-1])

        # specify the number of submovies to analyze
        self.num_submovies = (self.num_frames - roll_size) // roll_by

        # return the time-axis means for each channel
        ind = kern // 2
        self.means = nd.uniform_filter(self.image[:,0,:,:,:], size = (1,1,kern,kern))[:,:,ind:-ind:step, ind:-ind:step]
        self.xpix = self.means.shape[-2]
        self.ypix = self.means.shape[-1]
        self.num_boxes = self.xpix*self.ypix
        self.means = self.means.reshape(self.means.shape[0], self.means.shape[1], self.num_boxes)

    # function to return the autocorrelation of each box in the image stack for each channel
    def calc_ACF(self, peak_thresh=0.1):
        '''
        
        '''
    
        # make empty arrays to populate with 1) period measurements and 2) acf curves
        self.periods = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))
        self.acfs = np.zeros(shape = (self.num_submovies, self.num_channels, self.num_boxes, self.roll_size*2-1))

        with tqdm(total = self.num_submovies*self.num_channels*self.xpix*self.ypix) as pbar:
            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for box in range(self.num_boxes):
                        pbar.update(1)
                        # calculate full autocorrelation
                        signal = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, channel, box]
                        corr_signal = signal - signal.mean()
                        acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
                        # normalize the curve
                        acf_curve = acf_curve / (self.roll_size * signal.std() ** 2)
                        peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
                        # absolute difference between each peak and zero
                        peaks_abs = abs(peaks - acf_curve.shape[0]//2)
                        # if peaks were identified, pick the one closest to the center
                        if len(peaks) > 1:
                            delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
                        # otherwise, return nans for both period and autocorrelation curve
                        else:
                            delay = np.nan
                            acf_curve = np.full((self.roll_size*2-1), np.nan)
                        self.periods[submovie, channel, box] = delay
                        self.acfs[submovie, channel, box] = acf_curve

        return self.periods, self.acfs

    def calc_CCF(self):
        '''
        
        '''
        # make a list of unique channel combinations to calculate CCF for
        channels = list(range(self.num_channels))
        self.channel_combos = []
        for i in range(self.num_channels):
            for j in channels[i+1:]:
                self.channel_combos.append([channels[i],j])
        num_combos = len(self.channel_combos)

        # make empty arrays to populate with 1) period measurements and 2) acf curves   
        self.shifts = np.zeros(shape=(self.num_submovies, num_combos, self.num_boxes))
        self.ccfs = np.zeros(shape=(self.num_submovies, num_combos, self.num_boxes, self.roll_size*2-1))

        with tqdm(total=self.num_submovies*num_combos*self.num_boxes) as pbar:
            for submovie in range(self.num_submovies):
                for combo_number, combo in enumerate(self.channel_combos):
                    for box in range(self.num_boxes):
                        pbar.update(1)

                        # calculate full cross-correlation
                        signal1 = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[0], box]
                        signal2 = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[1], box]
                        corr_signal1 = signal1 - signal1.mean()
                        corr_signal2 = signal2 - signal2.mean()
                        cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
                        # normalize the curve
                        cc_curve = cc_curve / (self.roll_size * signal1.std() * signal2.std())
                        peaks, _ = sig.find_peaks(cc_curve, prominence=0.1)
                        # absolute difference between each peak and zero
                        peaks_abs = abs(peaks - cc_curve.shape[0]//2)
                        # if peaks were identified, pick the one closest to the center
                        if len(peaks) > 1:
                            delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
                        # otherwise, return nans for both period and autocorrelation curve
                        else:
                            delay = np.nan
                            cc_curve = np.full((self.roll_size*2-1), np.nan)
                        self.shifts[submovie, combo_number, box] = delay
                        self.ccfs[submovie, combo_number, box] = cc_curve

        return self.shifts, self.ccfs

    def calc_peak_props(self):
        '''
        
        '''
        # make empty arrays to fill with peak measurements for each channel
        self.peak_widths = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))
        self.peak_maxs = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))
        self.peak_mins = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))

        with tqdm(total = self.num_submovies*self.num_channels*self.xpix*self.ypix) as pbar:
            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for box in range(self.num_boxes):
                        pbar.update(1)

                        signal = sig.savgol_filter(self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, channel, box], window_length=11, polyorder=2)
                        peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

                        # if peaks detected, calculate properties and return property averages. Otherwise return nans
                        if len(peaks) > 0:
                            proms, _, _ = sig.peak_prominences(signal, peaks)
                            widths, _, _, _ = sig.peak_widths(signal, peaks, rel_height=0.5)
                            mean_width = np.mean(widths, axis=0)
                            mean_max = np.mean(signal[peaks], axis = 0)
                            mean_min = np.mean(signal[peaks]-proms, axis = 0)
                            self.peak_widths[submovie, channel, box] = mean_width
                            self.peak_maxs[submovie, channel, box] = mean_max
                            self.peak_mins[submovie, channel, box] = mean_min
                        else:
                            self.peak_widths[submovie, channel, box] = np.nan
                            self.peak_maxs[submovie, channel, box] = np.nan
                            self.peak_mins[submovie, channel, box] = np.nan   

        self.peak_amps = self.peak_maxs - self.peak_mins
        self.peak_rel_amps = self.peak_amps / self.peak_mins

        return self.peak_widths, self.peak_maxs, self.peak_mins, self.peak_amps, self.peak_rel_amps

    # function to summarize the results in the acf_results, ccf_results, and peak_results dictionaries as a dataframe
    def get_submovie_measurements(self):
        '''
        Gathers period, shift, and peak properties measurments (if they exist), appends some simple statistics, 
        and returns a SEPARATE dataframe with raw and summarized measurments for each submovie in the dataset.
        '''
        
        # function to summarize measurments statistics by appending them to the beginning of the measurement list
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
        col_names.extend([f'Box{i}' for i in range(self.num_boxes)])
        
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
                submovie_widths_with_stats = add_stats(self.peak_widths[submovie], 'Peak Width')
                submovie_maxs_with_stats = add_stats(self.peak_maxs[submovie], 'Peak Max')
                submovie_mins_with_stats = add_stats(self.peak_mins[submovie], 'Peak Min')
                submovie_amps_with_stats = add_stats(self.peak_amps[submovie], 'Peak Amp')
                submovie_rel_amps_with_stats = add_stats(self.peak_rel_amps[submovie], 'Peak Rel Amp')
                for channel in range(self.num_channels):
                    statified_measurements.append(submovie_widths_with_stats[channel])
                    statified_measurements.append(submovie_maxs_with_stats[channel])
                    statified_measurements.append(submovie_mins_with_stats[channel])
                    statified_measurements.append(submovie_amps_with_stats[channel])
                    statified_measurements.append(submovie_rel_amps_with_stats[channel])

            submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
            self.submovie_measurements.append(submovie_meas_df)

        return self.submovie_measurements

    def summarize_file(self):
        '''
        Summarizes the results of period, shift (if applicable) and peak analyses. Returns a
        SINGLE dataframe summarizing each of the relevant measurements for each submovie.
        '''
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
                    pcnt_no_period = (np.count_nonzero(np.isnan(self.periods[submovie, channel])) / self.num_boxes) * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(self.periods[submovie, channel])

            if hasattr(self, 'shifts'):
                for combo_number, combo in enumerate(self.channel_combos):
                    pcnt_no_shift = np.count_nonzero(np.isnan(self.ccfs[submovie, combo_number])) / self.num_boxes * 100
                    submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(self.shifts[submovie, combo_number])

            if hasattr(self, 'peak_widths'):
                for channel in range(self.num_channels):
                    # using widths, but because these are all assigned together it applies to all peak properties
                    pcnt_no_peaks = np.count_nonzero(np.isnan(self.peak_widths[submovie, channel])) / self.num_boxes * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Width'] = func(self.peak_widths[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Max'] = func(self.peak_maxs[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Min'] = func(self.peak_mins[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Amp'] = func(self.peak_amps[submovie, channel])
            all_submovie_summary.append(submovie_summary)
        
        col_names = [key for key in all_submovie_summary[0].keys()]
        self.full_movie_summary = pd.DataFrame(all_submovie_summary, columns = col_names)
                
        return self.full_movie_summary


    # function to plot the date from the self.dile_data_summary dataframe
    def plot_rolling_summary(self):
        '''
        This function plots the data from the self.full_movie_summary dataframe.
        '''
        def return_plot(independent_variable, dependent_variable, dependent_error, y_label):
            '''
            This function returns plot objects to its parent fuction
            '''                
            fig, ax = plt.subplots()
            # plot the dataframe
            ax.plot(self.full_movie_summary[independent_variable], 
                         self.full_movie_summary[dependent_variable])
            # fill between the ± standard deviatio of the dependent variable
            ax.fill_between(x = self.full_movie_summary[independent_variable],
                            y1 = self.full_movie_summary[dependent_variable] - self.full_movie_summary[dependent_error],
                            y2 = self.full_movie_summary[dependent_variable] + self.full_movie_summary[dependent_error],
                            color = 'blue',
                            alpha = 0.25)

            ax.set_xlabel('Frame Number')
            ax.set_ylabel(y_label)
            ax.set_title(f'{y_label} over time')
            plt.close(fig)
            return fig

        # empty list to fill with plots
        self.plot_list = {}
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
                self.plot_list[f'Ch {channel + 1} Peak Width'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Width',
                                                                            f'Ch {channel + 1} StdDev Peak Width',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Width (frames)')
                self.plot_list[f'Ch {channel + 1} Peak Max'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Max',
                                                                            f'Ch {channel + 1} StdDev Peak Max',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Max (frames)')
                self.plot_list[f'Ch {channel + 1} Peak Min'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Min',
                                                                            f'Ch {channel + 1} StdDev Peak Min',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Min (frames)')
                self.plot_list[f'Ch {channel + 1} Peak Amp'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Amp',
                                                                            f'Ch {channel + 1} StdDev Peak Amp',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Amp (frames)')    

        return self.plot_list