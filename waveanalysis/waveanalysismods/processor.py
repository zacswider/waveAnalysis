import scipy
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
        Calculate the autocorrelation of each box in the mean image stack for each channel.
        The period is calculated by find the first non-zero peak in the autocorrelation curve 
        that has a higher prominence than the peak_thresh parameter.

        Returns two objects: 
        self.periods is an array of shape (channels, boxes) containing the period of each box (units = frames)
        self.acfs is an array of shape (channels, boxes, frames*2-1) containing the autocorrelation curve for each box
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
        Calculate the crosscorrelation of each box in the mean image stack for each unique combinations
        of channels. The shift between signals is calculated by find the first non-zero peak in the 
        crosscorrelation curve that has a higher prominence than 10% of the range of curve values.
        
        Returns two objects: 
        self.shifts is an array of shape (channel combinations, boxes) containing the period of each box (units = frames)
        self.ccfs is an array of shape (channel combinations, boxes, frames*2-1) containing the crosscorrelation curve for each box
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
        # make a dictionary to store the arrays and measurments generated by this function so they don't have to be re-calculated later

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
                    delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                    delayIndex = peaks[delay]
                    delay_frames = delayIndex - cc_curve.shape[0]//2
                # otherwise, return nans for both period and autocorrelation curve
                else:
                    delay_frames = np.nan
                    cc_curve = np.full((self.num_frames*2-1), np.nan)
                self.shifts[combo_number, box] = delay_frames
                self.ccfs[combo_number, box] = cc_curve

        return self.shifts, self.ccfs

    def calc_peak_props(self):
        '''
        Calculate the peak properties of each box in the mean image stack for each channel.
        The signal within each box is smoothed using a Savitzky-Golay filter with a window size 11
        and polynomial order 2. The peak properties are calculated by finding peaks in the smoothed
        signal, and then calculating the properties of each peak. Returns the average of all peaks
        in each box.

        Returns five objects.. 
        self.peak_widths array of shape (channels, boxes) containing of shape (channels, boxes) containing the average peak width of each box in each channel (units = frames)
        self.peak_maxs array of shape (channels, boxes) containing the average peak height of each box in each channel (units = gray values)
        self.peak_mins array of shape (channels, boxes) containing the average peak trough of each box in each channel (units = gray values)
        self.peak_amps array of shape (channels, boxes) containing the average peak amplitude of each box in each channel (max - min; units = gray values)
        self.peak_rel_amps array of shape (channels, boxes) containing the average relative peak height of each box in each channel ((max - min) / min; units = gray values)
        self.ind_peak_props is a dictionary containing the smoothed signal, peak locations, maxs, mins, and widths for each box in each channel
        '''
        # make empty arrays to fill with peak measurements for each channel
        self.peak_widths = np.zeros(shape=(self.num_channels, self.num_boxes))
        self.peak_maxs = np.zeros(shape=(self.num_channels, self.num_boxes))
        self.peak_mins = np.zeros(shape=(self.num_channels, self.num_boxes))
        # make a dictionary to store the arrays and measurments generated by this function so they don't have to be re-calculated later
        self.ind_peak_props = {}
        
        for channel in range(self.num_channels):
            for box_num in range(self.num_boxes):

                signal = sig.savgol_filter(self.means[:,channel, box_num], window_length = 11, polyorder = 2)
                peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

                # if peaks detected, calculate properties and return property averages. Otherwise return nans
                if len(peaks) > 0:
                    proms, _, _ = sig.peak_prominences(signal, peaks)
                    widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
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
                
                # store the smoothed signal, peak locations, maxs, mins, and widths for each box in each channel
                self.ind_peak_props[f'Ch {channel} Box {box_num}'] = {'smoothed': signal, 
                                                         'peaks': peaks,
                                                         'proms': proms, 
                                                         'heights': heights, 
                                                         'leftIndex': leftIndex, 
                                                         'rightIndex': rightIndex}

        self.peak_amps = self.peak_maxs - self.peak_mins
        self.peak_rel_amps = self.peak_amps / self.peak_mins

        return self.peak_widths, self.peak_maxs, self.peak_mins, self.peak_amps, self.peak_rel_amps, self.ind_peak_props

    # function to plot a summary of the period measurements
    def plot_mean_ACF(self):
        '''
        Plots the mean autocorrelation curve ± the standard deviation of the curve for each channel.
        Also plots a histogram and a boxplot showing the distribution of the measurements over all of the boxes measured.

        Returns:
        self.acf_figs is a dictionary object containing the plot names as keys and the figure objects as values. These can
        be easily visualized by or saved to a file using the key value as a file name.
        '''
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel: str):
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
            ax['A'].set_title(f'{channel} Mean Autocorrelation Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of period values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of period values')
            ax['C'].set_ylabel(f'Measured period (frames)')
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
                                                                         f'Ch{channel + 1}')        

        return self.cf_figs

    # function to plot a summary of the period measurements
    def plot_mean_CCF(self):
        '''
        Plots the mean cross correlation curve ± the standard deviation of the curve for each channel combination.
        Also plots a histogram and a boxplot showing the distribution of the measurements over all of the boxes measured.

        Returns:
        self.ccf_figs is a dictionary object containing the plot names as keys and the figure objects as values. These can
        be easily visualized by or saved to a file using the key value as a file name.
        '''
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel_combo: str):
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
            ax['A'].set_title(f'{channel_combo} Mean Crosscorrelation Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of shift values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of shift values')
            ax['C'].set_ylabel(f'Measured shift (frames)')
            fig.subplots_adjust(hspace=0.25, wspace=0.5)   
            plt.close(fig)
            return fig

        # empty dict to fill with figures, in the event that we make more than one
        self.ccf_figs = {}
               
        if hasattr(self, 'ccfs'):
            if self.num_channels > 1:
                for combo_number, combo in enumerate(self.channel_combos):
                    self.ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_figure(self.ccfs[combo_number], 
                                                                                                self.shifts[combo_number], 
                                                                                                f'Ch{combo[0] + 1}-Ch{combo[1] + 1}')

        return self.ccf_figs

    def plot_mean_peak_props(self):
        '''
        Plots the distribution of peak properties for each channel.

        Returns:
        self.peak_figs is a dictionary object containing the plot names as keys and the figure objects as values. These can
        be easily visualized by or saved to a file using the key value as a file name.
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

    def plot_ind_acfs(self):
        '''
        Plot the raw signal and individual autocorrelation curve for each box in each channel. Annotates the first peak
        identified to estimate the period. 
        '''
        def return_figure(raw_signal: np.ndarray, acf_curve: np.ndarray, Ch_name: str, period: int):
            '''
            space saving function to generate individual plots with variable input
            '''
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(raw_signal)
            ax1.set_xlabel(f'{Ch_name} Raw Signal')
            ax1.set_ylabel('Mean box px value')
            ax2.plot(np.arange(-self.num_frames + 1, self.num_frames), acf_curve)
            ax2.set_ylabel('Autocorrelation')
            
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

        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_acf_plots = {}

        its = self.num_channels*self.num_boxes
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('ind acfs')
            for channel in range(self.num_channels):
                for box in range(self.num_boxes):
                    pbar.update(1)
                    self.ind_acf_plots[f'Ch{channel + 1} Box{box + 1} ACF'] = return_figure(self.means[:,channel, box], 
                                                                                            self.acfs[channel, box], 
                                                                                            f'Ch{channel + 1}', 
                                                                                            self.periods[channel, box])
        return self.ind_acf_plots

    def plot_ind_ccfs(self):
        '''
        Plot the raw signals and corresponding crosscurve for each box in each unique channel combo. 
        Annotates the first peak identified to estimate the temporal shift between signals. 
        '''
        def return_figure(ch1: np.ndarray, ch2: np.ndarray, ccf_curve: np.ndarray, ch1_name: str, ch2_name: str, shift: int):
            '''
            Space saving function to generate individual plots with variable input. returns a figure object.
            '''
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(ch1, color = 'tab:blue', label = ch1_name)
            ax1.plot(ch2, color = 'tab:orange', label = ch2_name)
            ax1.set_xlabel('time (frames)')
            ax1.set_ylabel('Mean box px value')
            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax2.plot(np.arange(-self.num_frames + 1, self.num_frames), ccf_curve)
            ax2.set_ylabel('Crosscorrelation')
            
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
            '''
            Normalize between 0 and 1
            '''
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_ccf_plots = {}

        if self.num_channels > 1:
            its = len(self.channel_combos)*self.num_boxes
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind ccfs')
                for combo_number, combo in enumerate(self.channel_combos):
                    for box in range(self.num_boxes):
                        pbar.update(1)
                        self.ind_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Box{box + 1} CCF'] = return_figure(ch1 = normalize(self.means[:, combo[0], box]),
                                                                                                        ch2 = normalize(self.means[:, combo[1], box]),
                                                                                                        ccf_curve = self.ccfs[combo_number, box],
                                                                                                        ch1_name = f'Ch{combo[0] + 1}',
                                                                                                        ch2_name = f'Ch{combo[1] + 1}',
                                                                                                        shift = self.shifts[combo_number, box])
        
        return self.ind_ccf_plots

    def plot_ind_peak_props(self):
        '''
        Plots the individual peaks measured with annotated peak properties for each box in each channel.

        Returns:
        self.peak_figs is a dictionary object containing the plot names as keys and the figure objects as values. These can
        be easily visualized by or saved to a file using the key value as a file name.
        '''
        def return_figure(box_signal: np.ndarray, prop_dict: dict, Ch_name: str):

            smoothed_signal = prop_dict['smoothed']
            peaks = prop_dict['peaks']
            proms = prop_dict['proms']
            heights = prop_dict['heights']
            leftIndex = prop_dict['leftIndex']
            rightIndex = prop_dict['rightIndex']

            fig, ax = plt.subplots()
            ax.plot(box_signal, color = 'tab:gray', label = 'raw signal')
            ax.plot(smoothed_signal, color = 'tab:cyan', label = 'smoothed signal')

            # plot all of the peak widths and amps in a loop
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
            # plot the first peak width and amp again so we can add it to the legend
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
            return fig

        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_peak_figs = {}
        # fill the dictionary with plots for each channel
        if hasattr(self, 'peak_widths'):
            its = self.num_channels*self.num_boxes
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind peaks')
                for channel in range(self.num_channels):
                    for box in range(self.num_boxes):
                        pbar.update(1)
                        self.ind_peak_figs[f'Ch{channel + 1} Box{box + 1} Peak Props'] = return_figure(self.means[:,channel, box],
                                                                                                    self.ind_peak_props[f'Ch {channel} Box {box}'],
                                                                                                    f'Ch{channel + 1} Box{box + 1}')

        return self.ind_peak_figs

    # function to summarize the results in the acf_results, ccf_results, and peak_results dictionaries as a dataframe
    def organize_measurements(self):
        '''
        Organizes the results of the ACF, CCF, and peak measurements into a dataframe. If any measurements were not
        performed, they will be excluded from the summary. Returns a dataframe with every measured parameter summarized
        by channel as well as the raw values measured for each box.

        Returns:
        self.im_measurements is a dataframe object containing the summarized results of the ACF, CCF, and peak measurements.
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
        Summarizes the results of all the measurements performed on the image.
        Returns dictionary object:
        self.file_data_summary contains the name of every summarized result for 
        each channel or channel combination as a key and the summarized results as a value.
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
        Calculate the autocorrelation of each box in the mean image stack for each submovie
        and each channel. The period is calculated by find the first non-zero peak in the 
        autocorrelation curve that has a higher prominence than the peak_thresh parameter.

        Returns two objects: 
        self.periods is an array of shape (submovies, channels, boxes) containing the period of each box (units = frames)
        self.acfs is an array of shape (submovies, channels, boxes, frames*2-1) containing the autocorrelation curve for each box
        '''
    
        # make empty arrays to populate with 1) period measurements and 2) acf curves
        self.periods = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))
        self.acfs = np.zeros(shape = (self.num_submovies, self.num_channels, self.num_boxes, self.roll_size*2-1))

        its = self.num_submovies*self.num_channels*self.xpix*self.ypix
        with tqdm(total = its, miniters=its/100) as pbar:
            pbar.set_description('Periods:')
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

        its = self.num_submovies*num_combos*self.num_boxes
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('Shifts:')
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
                            delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                            delayIndex = peaks[delay]
                            delay_frames = delayIndex - cc_curve.shape[0]//2
                        # otherwise, return nans for both period and autocorrelation curve
                        else:
                            delay_frames = np.nan
                            cc_curve = np.full((self.roll_size*2-1), np.nan)
                        self.shifts[submovie, combo_number, box] = delay_frames
                        self.ccfs[submovie, combo_number, box] = cc_curve

        return self.shifts, self.ccfs

    def calc_peak_props(self):
        '''
        Calculate the crosscorrelation of each unique channel combination in the mean image 
        stack for each submovie and each channel. The period is calculated by find the first
        non-zero peak in the crosscorrelation curve that has a higher prominence than the peak_thresh parameter.

        Returns two objects: 
        self.shifts is an array of shape (submovies, channel combos, boxes) containing the shift of each box (units = frames)
        self.ccfs is an array of shape (submovies, channel combos, boxes, frames*2-1) containing the crosscorrelation curve for each box
        '''
        # make empty arrays to fill with peak measurements for each channel
        self.peak_widths = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))
        self.peak_maxs = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))
        self.peak_mins = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes))

        its = self.num_submovies*self.num_channels*self.xpix*self.ypix
        with tqdm(total = its, miniters=its/100) as pbar:
            pbar.set_description('Peak props:')
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

        Returns:
        self.full_movie_summary is a dataframe summarizing the results of period, shift, and peak analyses for each submovie
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


    # function to plot the date from the self.file_data_summary dataframe
    def plot_rolling_summary(self):
        '''
        This function plots the data from the self.full_movie_summary dataframe.

        Returns:
        self.plot_list is a dictionary containing the names of the summary plots as keys and the fig object as values
        '''
        def return_plot(independent_variable, dependent_variable, dependent_error, y_label):
            '''
            This function returns plot objects to its parent function
            '''                
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

class KymographSignalProcessor: 
    def __init__(self, filename, im_save_path, img,line_width):
        self.filename = filename
        self.im_save_path = im_save_path
        self.img = img      
        self.line_width = line_width
        
        self.num_channels = self.img.shape[0]
        self.num_cols = self.img.shape[-1]
        self.num_rows = self.img.shape[-2]

        self.indv_line_values = self.calc_indv_line_values()

    def calc_indv_line_values(self, win_length=25):
        """
        Calculates the individual line values for each channel and column in the image data.

        Returns:
        - indv_line_values (numpy.ndarray): an array of individual line values for each channel and column
        """
        self.indv_line_values = np.zeros(shape=(self.num_channels, self.num_cols, self.num_rows))
        
        for channel in range(self.num_channels):
            for col_num in range(self.num_cols):
                if self.line_width == 1:
                    signal = scipy.signal.savgol_filter(self.img[channel, :, col_num], window_length = win_length, polyorder = 2)
                    self.indv_line_values[channel, col_num] = signal
                elif self.line_width % 2 != 0:
                    line_width_extra = int((self.line_width - 1) / 2)
                    if col_num + line_width_extra < self.num_cols and col_num - line_width_extra > -1:
                        signal = np.mean(self.img[channel, :, col_num-line_width_extra:col_num+line_width_extra], axis=1)
                        signal = scipy.signal.savgol_filter(signal, window_length = win_length, polyorder=2)
                        self.indv_line_values[channel, col_num] = signal

                    
        return self.indv_line_values

#################################
####### CALC INDV LINES #########
#################################

    def calc_ind_peak_props(self):
        """
        Calculate the peak properties for each channel and line in the provided data.

        This function generates smoothed signals for each line using a Savitzky-Golay filter, 
        and then detects peaks using the find_peaks function from the scipy.signal module. 
        Peak properties such as width, height, and prominence are then calculated and averaged for each line.

        The peak properties are stored in class attributes including ind_peak_widths, ind_peak_maxs, 
        ind_peak_mins, ind_peak_amps, and ind_peak_rel_amps. Additionally, a dictionary of peak-related 
        measurements for each channel and line is stored in the ind_peak_props attribute to avoid the 
        need for recalculation later.

        Returns:
        Tuple of numpy ndarrays and dictionary:
        - ind_peak_widths: numpy ndarray of shape (num_channels, num_cols) containing the average width of each peak for each channel and line.
        - ind_peak_maxs: numpy ndarray of shape (num_channels, num_cols) containing the average maximum value of each peak for each channel and line.
        - ind_peak_mins: numpy ndarray of shape (num_channels, num_cols) containing the average minimum value of each peak for each channel and line.
        - ind_peak_amps: numpy ndarray of shape (num_channels, num_cols) containing the average amplitude of each peak for each channel and line.
        - ind_peak_rel_amps: numpy ndarray of shape (num_channels, num_cols) containing the average relative amplitude of each peak for each channel and line.
        - ind_peak_props: dictionary containing the smoothed signal, peak locations, maxs, mins, and widths for each frame in each channel and line. Keys are in the form of Ch {channel} Line {line_num}.
        """
        # make empty arrays to fill with peak measurements for each channel
        self.ind_peak_widths = np.zeros(shape=(self.num_channels, self.num_cols))
        self.ind_peak_maxs = np.zeros(shape=(self.num_channels, self.num_cols))
        self.ind_peak_mins = np.zeros(shape=(self.num_channels, self.num_cols))
        # make a dictionary to store the arrays and measurements generated by this function so they don't have to be re-calculated later
        self.ind_peak_props = {}

        # generate the signals for each line, then find the peak
        for channel in range(self.num_channels):
            for line_num in range(self.num_cols):

                signal = scipy.signal.savgol_filter(self.indv_line_values[channel, line_num], window_length=11, polyorder=3)
                peaks, _ = scipy.signal.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*.6)

                # if peaks detected, calculate properties and return property averages. Otherwise return NaNs
                if len(peaks) > 0:
                    proms, _, _ = scipy.signal.peak_prominences(signal, peaks)
                    widths, heights, leftIndex, rightIndex = scipy.signal.peak_widths(signal, peaks, rel_height=0.5)
                    mean_width = np.mean(widths, axis=0)
                    mean_max = np.mean(signal[peaks], axis=0)
                    mean_min = np.mean(signal[peaks]-proms, axis=0)
                    self.ind_peak_widths[channel, line_num] = mean_width
                    self.ind_peak_maxs[channel, line_num] = mean_max
                    self.ind_peak_mins[channel, line_num] = mean_min

                    # store the smoothed signal, peak locations, maxs, mins, and widths for each frame in each channel
                    self.ind_peak_props[f'Ch {channel} Line {line_num}'] = {'smoothed': signal,
                                                                                                'peaks': peaks,
                                                                                                'proms': proms,
                                                                                                'heights': heights,
                                                                                                'leftIndex': leftIndex,
                                                                                                'rightIndex': rightIndex}

                else:
                    self.ind_peak_widths[channel, line_num] = np.nan
                    self.ind_peak_maxs[channel, line_num] = np.nan
                    self.ind_peak_mins[channel, line_num] = np.nan

                    self.ind_peak_props[f'Ch {channel} Line {line_num}'] = {'smoothed': np.nan,
                                                                                                'peaks': np.nan,
                                                                                                'proms': np.nan,
                                                                                                'heights': np.nan,
                                                                                                'leftIndex': np.nan,
                                                                                                'rightIndex': np.nan}

            # calculate amplitude and relative amplitude
            self.ind_peak_amps = self.ind_peak_maxs - self.ind_peak_mins
            self.ind_peak_rel_amps = self.ind_peak_amps / self.ind_peak_mins

        return self.ind_peak_widths, self.ind_peak_maxs, self.ind_peak_mins, self.ind_peak_amps, self.ind_peak_rel_amps, self.ind_peak_props

    def calc_indv_ACF(self, peak_thresh=0.1):
        """
        Calculates the autocorrelation functions (ACFs) and the periods of the ACFs for 
        each channel and line in the input data.

        Parameters:
            peak_thresh (float): The threshold for prominence in peak detection. Default is 0.1.

        Returns:
            Tuple containing the periods and ACFs for each channel and line. The periods are a 
            numpy array of shape (num_channels, num_cols), and the ACFs are a numpy array of 
            shape (num_channels, num_cols, num_rows*2-1).
        """
        # make empty arrays to populate with 1) period measurements and 2) acf curves
        self.periods = np.zeros(shape=(self.num_channels, self.num_cols))
        self.acfs = np.zeros(shape=(self.num_channels, self.num_cols, self.num_rows*2-1))

        for channel in range(self.num_channels):
            for col_num in range(self.num_cols):
                # calculate full autocorrelation
                signal = self.indv_line_values[channel, col_num, :]
                corr_signal = signal - signal.mean()
                acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
                # normalize the curve
                acf_curve = acf_curve / (self.num_rows * signal.std() ** 2)
                peaks, _ = scipy.signal.find_peaks(acf_curve, prominence=peak_thresh)
                # absolute difference between each peak and zero
                peaks_abs = abs(peaks - acf_curve.shape[0]//2)
                # if peaks were identified, pick the one closest to the center
                if len(peaks) > 1:
                    delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
                # otherwise, return nans for both period and autocorrelation curve
                else:
                    delay = np.nan
                    acf_curve = np.full((self.num_rows*2-1), np.nan)
                self.periods[channel, col_num] = delay
                self.acfs[channel, col_num] = acf_curve

        return self.periods, self.acfs

    def calc_indv_CCFs(self):
        '''
        This function calculates the cross-correlation function (CCF) for pairs of channels in 
        the dataset. It first creates a list of all possible combinations of channels and then 
        calculates the CCF for each combination across all columns in the dataset.

        The function initializes two arrays indv_shifts and indv_ccfs to store the shift and 
        CCF values respectively. The CCF is calculated using the calc_shifts() function.

        The function returns the indv_shifts and indv_ccfs arrays along with the list of channel 
        combinations channel_combos.
        '''
        # make a list of unique channel combinations to calculate CCF for
        channels = list(range(self.num_channels))
        self.channel_combos = []
        for i in range(self.num_channels):
            for j in channels[i + 1:]:
                self.channel_combos.append([channels[i], j])
        self.num_combos = len(self.channel_combos)

        # calc shifts and cross-corelation
        self.indv_shifts = np.zeros(shape=(self.num_combos, self.num_cols))
        self.indv_ccfs = np.zeros(shape=(self.num_combos, self.num_cols, self.num_rows*2-1))

        #for each channel combo, calculate the CCF
        for combo_number, combo in enumerate(self.channel_combos):
            for line_num in range(self.num_cols):
                signal1 = self.indv_line_values[combo[0], line_num]
                signal2 = self.indv_line_values[combo[1], line_num]

                delay_frames, cc_curve = self.calc_shifts(signal1, signal2, prominence=0.1)

                # The script has issues when the shift is very small or none, so minus the average period from the two channels
                period = (self.periods[combo[0],line_num] + self.periods[combo[1],line_num]) / 2
                if abs(delay_frames) > abs(period * .5):
                    if delay_frames < 0:
                        delay_frames = delay_frames + period
                    elif delay_frames > 0:
                        delay_frames = delay_frames - period

                self.indv_shifts[combo_number, line_num] = delay_frames
                self.indv_ccfs[combo_number, line_num] = cc_curve

        return self.indv_shifts, self.indv_ccfs, self.channel_combos

    def calc_shifts(self, signal1, signal2, prominence):
        """
        Calculates the cross-correlation between two signals and the time delay
        between them at which the correlation is highest.

        Parameters:
        signal1 : array-like
            First signal to correlate.
        signal2 : array-like
            Second signal to correlate.
        prominence : float
            Minimum prominence of peaks in the cross-correlation curve.

        Returns:
        delay_frames : float
            Number of frames by which signal2 is delayed with respect to signal1,
            so that their cross-correlation is maximized. NaN if either signal has
            no prominent peaks.
        cc_curve : numpy array
            Cross-correlation curve between signal1 and signal2.
            NaN if either signal has no prominent peaks.
        """
        #smooth signals
        signal1 = scipy.signal.savgol_filter(signal1, window_length=11, polyorder=3)
        signal2 = scipy.signal.savgol_filter(signal2, window_length=11, polyorder=3)

        # Find peaks in the signals
        peaks1, _ = scipy.signal.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
        peaks2, _ = scipy.signal.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

        if len(peaks1) > 0 and len(peaks2) > 0:
            corr_signal1 = signal1 - signal1.mean()
            corr_signal2 = signal2 - signal2.mean()
            cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
            # smooth the curve
            cc_curve = scipy.signal.savgol_filter(cc_curve, window_length=11, polyorder=3)
            # normalize the curve
            cc_curve = cc_curve / (self.num_rows * signal1.std() * signal2.std())
            # find peaks
            peaks, _ = scipy.signal.find_peaks(cc_curve, prominence=prominence)
            # absolute difference between each peak and zero
            peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
            # if peaks were identified, pick the one closest to the center
            if len(peaks) > 1:
                delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                delayIndex = peaks[delay]
                delay_frames = delayIndex - cc_curve.shape[0] // 2
            # otherwise, return NaNs for both period and autocorrelation curve
            else:
                delay_frames = np.nan
                cc_curve = np.full((self.num_rows * 2 - 1), np.nan)
        else:
            delay_frames = np.nan
            cc_curve = np.full((self.num_rows * 2 - 1), np.nan)

        return delay_frames, cc_curve

#################################
####### PLOT INDV LINES #########
#################################

    def plot_ind_peak_props(self):
        """
        Plots the peak properties of each individual peak found in the data.

        Returns:
            dict: A dictionary of figures, where each key is a string representing the 
            channel and line of the peak properties plotted, and the value is the 
            corresponding matplotlib figure.
        """
        def return_figure(line_signal: np.ndarray, prop_dict: dict, Ch_name: str):
            '''
            The return_figure function takes in a line signal array, a dictionary containing peak properties, and a string representing the name of the channel. It plots the raw signal and smoothed signal, along with vertical and horizontal lines indicating the positions of peaks, their heights, and their full width at half maximum (FWHM). The function returns the resulting figure object.

            Parameters:
                line_signal (np.ndarray): 1D array of signal values for a single channel and line.
                prop_dict (dict): Dictionary containing peak properties for the given channel and line, including the smoothed signal, peak indices, peak heights, FWHM, and left and right indices.
                Ch_name (str): Name of the channel, used for the plot title.

            Returns:
                fig (matplotlib.figure.Figure): Figure object containing the plotted signal and peak properties.
            '''
            smoothed_signal = prop_dict['smoothed']
            peaks = prop_dict['peaks']
            proms = prop_dict['proms']
            heights = prop_dict['heights']
            leftIndex = prop_dict['leftIndex']
            rightIndex = prop_dict['rightIndex']

            fig, ax = plt.subplots()
            ax.plot(line_signal, color = 'tab:gray', label = 'raw signal')
            ax.plot(smoothed_signal, color = 'tab:cyan', label = 'smoothed signal')

            # plot all of the peak widths and amps in a loop
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
                # plot the first peak width and amp again so we can add it to the legend
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

            return fig

        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_peak_figs = {}
        # fill the dictionary with plots for each channel
        if hasattr(self, 'ind_peak_widths'):
            its = self.num_channels*self.num_cols
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind peaks')
                for channel in range(self.num_channels):
                    for line in range(self.num_cols):
                        pbar.update(1)
                        self.ind_peak_figs[f'Ch {channel + 1 } Line {line + 1} Peak Props'] = return_figure(self.indv_line_values[channel, line, :],
                                                                                                          self.ind_peak_props[f'Ch {channel} Line {line}'],
                                                                                                          f'Ch {channel + 1} Line {line + 1}')

        return self.ind_peak_figs

    def plot_ind_acfs(self):
        """
        Plot individual autocorrelation functions (ACFs) for each channel and line in an image.
        
        Returns
        - dict: A dictionary containing figures for each channel and line, showing the raw signal and ACF, 
                with any identified periodicity highlighted.
        """
        def return_figure(raw_signal: np.ndarray, acf_curve: np.ndarray, Ch_name: str, period: int):
            """
            Plots the raw signal and autocorrelation function (ACF) curve of a given channel and line.

            Args:
            - raw_signal (np.ndarray): 1D array of raw signal values for the given channel and line.
            - acf_curve (np.ndarray): 1D array of ACF values for the given channel and line.
            - Ch_name (str): Name of the channel being plotted.
            - period (int): Period (in frames) of the signal, if identified. If no period is identified, should be np.nan.

            Returns:
            - fig (matplotlib.figure.Figure): Figure object containing the plotted raw signal and ACF curve.
            """
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(raw_signal)
            ax1.set_xlabel(f'{Ch_name} Raw Signal')
            ax1.set_ylabel('Mean line px value')
            ax2.plot(np.arange(-self.num_rows + 1, self.num_rows), acf_curve)
            ax2.set_ylabel('Autocorrelation')
            
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

        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_acf_plots = {}

        its = self.num_channels*self.num_cols
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('ind acfs')
            for channel in range(self.num_channels):
                for line in range(self.num_cols):
                    pbar.update(1)
                    self.ind_acf_plots[f'Ch{channel + 1} Line {line + 1} ACF'] = return_figure(self.indv_line_values[channel, line, :], 
                                                                                            self.acfs[channel, line], 
                                                                                            f'Ch{channel + 1}', 
                                                                                            self.periods[channel, line])
        return self.ind_acf_plots

    def plot_ind_ccfs(self):
        """
        Plots the individual cross-correlation functions (CCFs) between each pair of channels and for each row of the image.
        The resulting figures are saved in a dictionary, where the keys are formatted as 'Ch{channel1}-Ch{channel2} Line {line} CCF'.

        Returns:
        - ind_ccf_plots (dict): A dictionary containing the generated figures, where each key corresponds to a specific CCF plot.
        """
        def return_figure(ch1: np.ndarray, ch2: np.ndarray, ccf_curve: np.ndarray, ch1_name: str, ch2_name: str, shift: int):
            """
            Create a plot with two subplots showing the time series of two channels and their cross-correlation function (CCF).

            Args:
                ch1 (numpy.ndarray): Array of mean pixel values for channel 1.
                ch2 (numpy.ndarray): Array of mean pixel values for channel 2.
                ccf_curve (numpy.ndarray): Array of cross-correlation values.
                ch1_name (str): Name of channel 1.
                ch2_name (str): Name of channel 2.
                shift (int): Detected shift between channels, in number of frames.

            Returns:
                matplotlib.figure.Figure: The created figure object.

            """
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(ch1, color = 'tab:blue', label = ch1_name)
            ax1.plot(ch2, color = 'tab:orange', label = ch2_name)
            ax1.set_xlabel('time (frames)')
            ax1.set_ylabel('Mean line px value')
            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax2.plot(np.arange(-self.num_rows + 1, self.num_rows), ccf_curve)
            ax2.set_ylabel('Crosscorrelation')
            
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
            '''
            Normalize between 0 and 1
            '''
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_ccf_plots = {}

        if self.num_channels > 1:
            its = len(self.channel_combos)*self.num_cols
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind ccfs')
                for combo_number, combo in enumerate(self.channel_combos):
                    for line in range(self.num_cols):
                        pbar.update(1)
                        self.ind_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Line {line + 1} CCF'] = return_figure(ch1 = normalize(self.indv_line_values[combo[0], line, :]),
                                                                                                        ch2 = normalize(self.indv_line_values[combo[1], line, :]),
                                                                                                        ccf_curve = self.indv_ccfs[combo_number, line],
                                                                                                        ch1_name = f'Ch{combo[0] + 1}',
                                                                                                        ch2_name = f'Ch{combo[1] + 1}',
                                                                                                        shift = self.indv_shifts[combo_number, line])
        
        return self.ind_ccf_plots
    
#################################
####### PLOT MEAN LINES #########
#################################

    def plot_mean_ACF(self):
        """
        Generates and returns a dictionary of figures that display the mean autocorrelation curves, 
        histograms of the periods measured for each curve, and boxplots of the periods measured for 
        each curve, for each channel in the image sequence.
        
        Returns:
        self.cf_figs: a dictionary of figures containing mean autocorrelation curves, histograms of 
            periods, and boxplots of periods for each channel
        """
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel: str):
            """
            Returns a matplotlib figure containing a plot of the mean autocorrelation curve with standard deviation shading, 
            a histogram of the calculated period values, and a boxplot of the period values.
            
            Parameters:
            -----------
            arr : numpy.ndarray
                A 2D numpy array containing the autocorrelation curves for each frame.
            shifts_or_periods : numpy.ndarray
                A 1D numpy array containing the calculated periods or shifts between the two channels.
            channel : str
                A string specifying which channel the data belongs to.
                
            Returns:
            --------
            fig : matplotlib.figure.Figure
                A matplotlib figure containing the plotted data.
            """
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_rows + 1, self.num_rows)
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel} Mean Autocorrelation Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of period values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of period values')
            ax['C'].set_ylabel(f'Measured period (frames)')
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
                                                                         f'Ch{channel + 1}')        

        return self.cf_figs

    def plot_mean_CCF(self):
        """
        Plot the mean cross-correlation curve (CCF) for each channel combination, along with the standard deviation.

        This method uses the `return_figure` function to create a figure for each channel combination. The figure contains three subplots:
        - A: the mean CCF curve, calculated from the cross-correlation values for the corresponding channel combination. The standard deviation is shown as a shaded area around the curve.
        - B: a histogram of the shift values for the corresponding channel combination.
        - C: a boxplot of the shift values for the corresponding channel combination.

        Returns:
        ccf_figs: dict
            A dictionary containing the figures created for each channel combination. The keys are strings with the format 'ChX-ChY Mean CCF', where X and Y are the channel numbers (starting from 1).
        """
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel_combo: str):
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_rows + 1, self.num_rows)
            ax['A'].plot(x_axis, arr_mean, color='orange')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='orange', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel_combo} Mean Crosscorrelation Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of shift values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of shift values')
            ax['C'].set_ylabel(f'Measured shift (frames)')
            fig.subplots_adjust(hspace=0.25, wspace=0.5)   

                # Customize plot appearance
            fig.patch.set_facecolor('black')  # Set the background color to black
            for _, subplot in ax.items():
                subplot.set_facecolor('black')
                subplot.spines['top'].set_visible(False)
                subplot.spines['right'].set_visible(False)
                subplot.spines['bottom'].set_color('white')
                subplot.spines['left'].set_color('white')
                subplot.xaxis.label.set_color('white')
                subplot.yaxis.label.set_color('white')
                subplot.title.set_color('white')
                subplot.tick_params(axis='x', colors='white')
                subplot.tick_params(axis='y', colors='white')
            
            plt.close(fig)
            return fig

        # empty dict to fill with figures, in the event that we make more than one
        self.ccf_figs = {}
               
        if hasattr(self, 'indv_ccfs'):
            if self.num_channels > 1:
                for combo_number, combo in enumerate(self.channel_combos):
                    self.ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_figure(self.indv_ccfs[combo_number], 
                                                                                                self.indv_shifts[combo_number], 
                                                                                                f'Ch{combo[0] + 1}-Ch{combo[1] + 1}')

        return self.ccf_figs

    def plot_mean_peak_props(self):
        """
        Plots histograms and boxplots of peak properties across channels. 

        Returns a dictionary of figures for each channel, with keys in the format of 'Ch{channel_num} Peak Props'.

        The function takes the following arguments:
        - self: object of the class containing data for all channels
        """
        def return_figure(min_array: np.ndarray, max_array: np.ndarray, amp_array: np.ndarray, width_array: np.ndarray, Ch_name: str):
            """
            Generate a figure with four subplots displaying histograms and boxplots of the peak values and widths for a given channel.

            Parameters
            ----------
            min_array : numpy.ndarray
                1D array containing the minimum values of the detected peaks.
            max_array : numpy.ndarray
                1D array containing the maximum values of the detected peaks.
            amp_array : numpy.ndarray
                1D array containing the amplitude values of the detected peaks.
            width_array : numpy.ndarray
                1D array containing the width values of the detected peaks.
            Ch_name : str
                A string representing the name of the channel.

            Returns
            -------
            matplotlib.figure.Figure
                A Figure object containing four subplots displaying histograms and boxplots of the peak values and widths.

            Notes
            -----
            The function filters out any NaN values from the input arrays before generating the plots. The returned figure is not displayed, but instead closed before being returned.
            """
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
            lines = ax2.boxplot([val[0] for val in plot_params.values()], patch_artist = True)
            ax2.set_xticklabels(plot_params.keys())
            for line, line_color in zip(lines['boxes'], [val[1] for val in plot_params.values()]):
                line.set_color(line_color)

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
        if hasattr(self, 'ind_peak_widths'):
            for channel in range(self.num_channels):
                self.peak_figs[f'Ch{channel + 1} Peak Props'] = return_figure(self.ind_peak_mins[channel], 
                                                                              self.ind_peak_maxs[channel], 
                                                                              self.ind_peak_amps[channel], 
                                                                              self.ind_peak_widths[channel], 
                                                                              f'Ch{channel + 1}')

        return self.peak_figs

#################################
####### SAVING MEASUREMENTS #####
#################################

    def organize_measurements(self):
        """
        Summarize measurements statistics and combine them into a single pandas DataFrame.
    
        This method generates summary statistics for the different types of measurements
        performed on an image, including period, shift, peak width, peak maximum, peak minimum,
        peak amplitude, and peak relative amplitude. The summary statistics are computed as
        mean, median, standard deviation, and standard error of the mean, and are appended
        to the beginning of the list of individual measurements for each type. The resulting
        lists are combined into a single list of statified measurements, which is turned into
        a pandas DataFrame with columns for the parameter name, mean, median, standard deviation,
        standard error of the mean, and the measurements for each line of the image.
        
        Returns:
            A pandas DataFrame containing the summary statistics for all the measurements
            performed on the image.
        """
        # function to summarize measurements statistics by appending them to the beginning of the measurement list
        def add_stats(measurements: np.ndarray, measurement_name: str):
            """
            Adds statistical measures to the given measurements and returns a list of lists
            where each inner list contains the statistical measures along with the corresponding
            measurement values and name.

            Parameters:
                measurements (np.ndarray): An array of measurements to which statistics should be added.
                measurement_name (str): A string indicating the type of measurement being processed.

            Returns:
                List[List[Union[str, float]]]: A list of lists containing statistical measures along
                with corresponding measurement values and name.
            """
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

        # column names for the dataframe summarizing the line results
        col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        col_names.extend([f'Line {i}' for i in range(self.num_cols)])
        # combine all the statified measurements into a single list
        statified_measurements = []

        # insert Mean, Median, StdDev, and SEM into the beginning of each list
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

        # and turn it into a dataframe
        self.im_measurements = pd.DataFrame(statified_measurements, columns = col_names)
        return self.im_measurements

    def summarize_image(self, file_name = None, group_name = None):
        """
        Summarize the measurements of an image file.

        Args:
        file_name (str): the name of the image file to summarize (default: None).
        group_name (str): the name of the group the image belongs to (default: None).

        Returns:
        dict: a dictionary containing the summarized measurements for the image.

        The function summarizes the measurements of an image file by calculating various statistics for its periods, shifts, and peaks. The summarized measurements are stored in a dictionary with keys representing the measurement type and values representing the calculated statistics. The dictionary is returned at the end of the function.
        """
        # dictionary to store the summarized measurements for each image
        self.file_data_summary = {}
        
        if file_name:
            self.file_data_summary['File Name'] = file_name
        if group_name:
            self.file_data_summary['Group Name'] = group_name
        self.file_data_summary['Num Lines'] = self.num_cols

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
    
    def save_means_to_csv(self, main_save_path, group_names,summary_df):
        """
        Save the mean values of certain metrics to separate CSV files for each group.

        Args:
            main_save_path (str): The path where the CSV files will be saved.
            group_names (list): A list of strings representing the names of the groups to be analyzed.
            summary_df (pandas DataFrame): The summary DataFrame containing the data to be analyzed.
        """
        for channel in range(self.num_channels):
            data_to_extract = [f"Ch {channel + 1} {data}" for data in ['Mean Period', 'Mean Peak Width', 'Mean Peak Max', 'Mean Peak Min', 'Mean Peak Amp', 'Mean Peak Rel Amp']]

            # Set up the output file paths
            output_file_paths = {}
            for data_name in data_to_extract:
                output_file_paths[f"{data_name}"] = f"{main_save_path}/{data_name.lower().replace(' ', '_')}_means.csv"
            
            # extract all the data (data_to_extract) from the summary df and store in a data frame
            result_df = pd.DataFrame(columns=['Data Type', 'Group Name', 'Value'])
            for data in data_to_extract:
                for group_name in group_names:
                    subset_df = summary_df.loc[summary_df['File Name'].str.contains(group_name)]
                    values = subset_df[data].tolist()
                    new_df = pd.DataFrame({'Data Type': data, 'Group Name': group_name, 'Value': values})
                    result_df = pd.concat([result_df, new_df], ignore_index=True)

            # extract, sort, and save individual tables for each data type in data_to_extract
            for data_type, output_path in output_file_paths.items():
                table = result_df[result_df['Data Type'] == data_type][['Group Name', 'Value']]
                table = pd.pivot_table(table, index=table.index, columns='Group Name', values='Value')
                for col in table.columns:
                    table[col] = sorted(table[col], key=lambda x: 1 if pd.isna(x) or x == '' else 0)
                table.to_csv(output_path, index=False)