import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
from tifffile import imread, imwrite, TiffFile

class SignalProcessor:
    
    def __init__(self, image_path, box_size):
        self.image_path = image_path
        self.box_size = box_size
        self.image = imread(self.image_path)

        # standardize image dimensions
        with TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)
        self.num_slices = metadata.get('slices', 1)
        self.num_frames = metadata.get('frames', 1)
        print(f'Image dimensions before reshaping {self.image.shape}')
        print(f'number of items before reshaping is {self.image.size}')
        print(f'number of channels is {self.num_channels}')
        print(f'number of slices is {self.num_slices}')
        print(f'number of frames is {self.num_frames}')
        self.image = self.image.reshape(self.num_frames, 
                                        self.num_slices, 
                                        self.num_channels, 
                                        self.image.shape[-2], 
                                        self.image.shape[-1])
        print(f'Image dimensions after reshaping {self.image.shape}')
        print(f'number of items after reshaping is {self.image.size}')

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
            print(f'Image dimensions after projecting {self.image.shape}')
        
        # calculate number of boxes in each dimension
        self.x_dim = self.image.shape[-1]
        self.y_dim = self.image.shape[-2]
        self.x_boxes = self.x_dim // self.box_size
        self.y_boxes = self.y_dim // self.box_size
        self.num_boxes = self.x_boxes * self.y_boxes
        print('reload succcesful')
        # return the time-axis means for each channel
        self.box_means = np.zeros((self.x_boxes, self.y_boxes, self.num_channels, self.num_frames))
        for channel in range(self.num_channels):
            print(f'Calculating channel {channel+1}')
            for x in range(self.x_boxes):
                for y in range(self.y_boxes):
                    self.box_means[x, y, channel, :] = np.mean(self.image[:, 0, channel, (x*self.box_size):(x*self.box_size+self.box_size), (y*self.box_size):(y*self.box_size+self.box_size)], axis=(1,2))
        # reshape into 2D array. Shape is (channels, boxes, frames)
        print(f'Box means shape is {self.box_means.shape} bfore reshaping')
        self.box_means = self.box_means.reshape((self.num_boxes, self.num_channels, self.num_frames))
        print(f'Box means shape is {self.box_means.shape} after reshaping')

        # empty dictionary to fill with measurements. These will subsequently be populated by the functions
        # below and returned to the user. They will also be used by the summarizing and plotting functions.
        self.acf_results = {}
        self.ccf_results = {}
        self.peak_results = {}

    # function to return the autocorrelation of each box in the image stack for each channel
    def calc_ACF(self, peak_thresh):
        '''
        Returns a dictionary containing the channel identify and box number as keys and the
        calculated period and autocorrelation curve as a values in a tuple.
        '''
        for channel in range(self.num_channels):
            for box_num in range(self.num_boxes):
                # calculate full autocorrelation
                signal = self.box_means[box_num, channel]
                acf_curve = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')
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

                self.acf_results[f'Ch{channel+1}_ACF_box{box_num}'] = (delay, acf_curve)
        return self.acf_results
                
    # function to return the cross-correlation of each box in the image stack
    def calc_CCF(self):
        '''
        Returns a dictionary containing the box number as keys and the
        calculated shift and crosscorrelation curve as a values in a tuple.
        '''
        assert self.num_channels == 2, 'CCF only works for 2 channels'
        for box_num in range(self.num_boxes):
            # calculate full cross-correlation (channels, boxes, frames)
            signal_1 = self.box_means[box_num, 1, :]
            signal_2 = self.box_means[box_num, 0, :]
            cc_curve = np.correlate(signal_1 - signal_1.mean(), signal_2 - signal_2.mean(), mode='full')
            # normalize the curve
            cc_curve = cc_curve / (self.num_frames * signal_1.std() * signal_2.std())
            # find the peak closes to zero
            peaks, _ = sig.find_peaks(cc_curve)
            peaks_abs = abs(peaks - cc_curve.shape[0]//2)
            delay_index = peaks[np.argmin(peaks_abs)]
            shift = delay_index - cc_curve.shape[0]//2
            self.ccf_results[f'CCF_box{box_num}'] = (shift, cc_curve)
        return self.ccf_results

    # function to return the peak properties of each box for each channel
    def calc_peaks(self): 
        '''
        Returns a dictionary containing the channel identify and box number as keys and the
        calculated peak properties as a values in a tuple (width, max, min, amp, relAmp).
        '''
        for channel in range(self.num_channels):
            for box_num in range(self.num_boxes):
                signal = sig.savgol_filter(self.box_means[box_num, channel], window_length = 11, polyorder = 2)
                peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

                # if peaks detected, calculate properties and return property averages. Otherwise return nans
                if len(peaks) > 0:
                    proms, _, _ = sig.peak_prominences(signal, peaks)
                    widths, _, _, _ = sig.peak_widths(signal, peaks, rel_height=0.5)
                    mean_width = np.mean(widths, axis=0)
                    mean_max = np.mean(signal[peaks], axis = 0)
                    mean_min = np.mean(signal[peaks]-proms, axis = 0)
                    mean_amp = mean_max - mean_min
                    mean_rel_amp = mean_amp / mean_min
                    self.peak_results[f'Ch{channel+1}_box{box_num}'] = (mean_width, mean_max, mean_min, mean_amp, mean_rel_amp)
                else:
                    self.peak_results[f'Ch{channel+1}_box{box_num}'] = (np.nan, np.nan, np.nan, np.nan, np.nan)

        return self.peak_results

    # function to summarize measurments statistics by appending them to the beginning of the measurement list
    def add_stats(self, measurements: list, measurement_name: str):
        '''
        Accepts a list of measurements. Calculates the mean, median, standard deviation, and SEM,
        and append them to the beginning of the list in that order. Finally, appends the name of
        the measurement of the beginning of the list.
        '''
        meas_mean = np.mean(measurements)
        meas_median = np.median(measurements)
        meas_std = np.std(measurements)
        meas_sem = meas_std / np.sqrt(len(measurements))
        measurements.insert(0, meas_mean)
        measurements.insert(1, meas_median)
        measurements.insert(2, meas_std)
        measurements.insert(3, meas_sem)
        measurements.insert(0, measurement_name)
        return measurements

    # function to summarize the results in the acf_results, ccf_results, and peak_results dictionaries as a dataframe
    def summarize_results(self, file_name = None, group_name = None):
        '''
        Takes the results from the calc_ACF, calc_CCF, and calc_peaks functions and returns a dataframe.
        '''
        # initial column names
        col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        for box in range(self.num_boxes):
            # add box number to column names
            col_names.append(f"Box {box}")

        # initialize lists to fill with measurements for each box and summary statistics
        if len(self.acf_results) > 0:
            ch1_period_measurements = []
            for key, value in self.acf_results.items():
                if 'Ch1' in key:
                    ch1_period_measurements.append(value[0])
            # calculate the number of boxes that didn't return a period
            ch1_pcnt_no_period = ((self.num_boxes-len(ch1_period_measurements))/self.num_boxes)*100

            if self.num_channels == 2:
                ch2_period_measurements = []
                for key, value in self.acf_results.items():
                    if 'Ch2' in key:
                        ch2_period_measurements.append(value[0])
                # calculate the number of boxes that didn't return a period
                ch2_pcnt_no_period = ((self.num_boxes-len(ch2_period_measurements))/self.num_boxes)*100
        
        if len(self.ccf_results) > 0:
            shift_measurements = []
            for key, value in self.ccf_results.items():
                shift_measurements.append(value[0])

        if len(self.peak_results) > 0:
            ch1_width_measurements = []
            ch1_max_measurements = []
            ch1_min_measurements = []
            ch1_amp_measurements = []
            ch1_relAmp_measurements = []
            for key, value in self.peak_results.items():
                if 'Ch1' in key:
                    ch1_width_measurements.append(value[0])
                    ch1_max_measurements.append(value[1])
                    ch1_min_measurements.append(value[2])
                    ch1_amp_measurements.append(value[3])
                    ch1_relAmp_measurements.append(value[4])
            if self.num_channels == 2:
                ch2_width_measurements = []
                ch2_max_measurements = []
                ch2_min_measurements = []
                ch2_amp_measurements = []
                ch2_relAmp_measurements = []
                for key, value in self.peak_results.items():
                    if 'Ch2' in key:
                        ch2_width_measurements.append(value[0])
                        ch2_max_measurements.append(value[1])
                        ch2_min_measurements.append(value[2])
                        ch2_amp_measurements.append(value[3])
                        ch2_relAmp_measurements.append(value[4])

        # insert Mean, Median, StdDev, and SEM into the beginning of each  list
        if len(self.acf_results) > 0:
            ch1_period_measurements = self.add_stats(ch1_period_measurements, "Ch1 Period")
            if self.num_channels == 2:
                ch2_period_measurements = self.add_stats(ch2_period_measurements, "Ch2 Period")
        
        if len(self.ccf_results) > 0:
            shift_measurements = self.add_stats(shift_measurements, "Shift")

        if len(self.peak_results) > 0:
            ch1_amp_measurements = self.add_stats(ch1_amp_measurements, "Ch1 Amplitude")
            ch1_width_measurements = self.add_stats(ch1_width_measurements, "Ch1 Width")
            ch1_max_measurements = self.add_stats(ch1_max_measurements, "Ch1 Max")
            ch1_min_measurements = self.add_stats(ch1_min_measurements, "Ch1 Min")
            ch1_relAmp_measurements = self.add_stats(ch1_relAmp_measurements, "Ch1 Relative Amplitude")
            if self.num_channels == 2:
                ch2_amp_measurements = self.add_stats(ch2_amp_measurements, "Ch2 Amplitude")
                ch2_width_measurements = self.add_stats(ch2_width_measurements, "Ch2 Width")
                ch2_max_measurements = self.add_stats(ch2_max_measurements, "Ch2 Max")
                ch2_min_measurements = self.add_stats(ch2_min_measurements, "Ch2 Min")
                ch2_relAmp_measurements = self.add_stats(ch2_relAmp_measurements, "Ch2 Relative Amplitude")

        # append the lists to the dictionary, if they exist
        self.im_measurements = pd.DataFrame(columns = col_names)
        if len(self.acf_results) > 0:
            self.im_measurements = pd.concat([self.im_measurements, pd.DataFrame([ch1_period_measurements], columns = col_names)], axis = 0)
            if self.num_channels == 2:
                self.im_measurements = pd.concat([self.im_measurements, pd.DataFrame([ch2_period_measurements], columns = col_names)], axis = 0)
        if len(self.ccf_results) > 0:
            self.im_measurements = pd.concat([self.im_measurements, pd.DataFrame([shift_measurements], columns = col_names)], axis = 0)
        if len(self.peak_results) > 0:
            self.im_measurements = pd.concat([self.im_measurements, pd.DataFrame(data=[ch1_amp_measurements,
                                                                                       ch1_width_measurements,
                                                                                       ch1_max_measurements,
                                                                                       ch1_min_measurements,
                                                                                       ch1_relAmp_measurements], columns = col_names)], axis = 0)
            if self.num_channels == 2:
                self.im_measurements = pd.concat([self.im_measurements, pd.DataFrame(data=[ch2_amp_measurements,
                                                                                       ch2_width_measurements,
                                                                                       ch2_max_measurements,
                                                                                       ch2_min_measurements,
                                                                                       ch2_relAmp_measurements], columns = col_names)], axis = 0)
 
        # empty dictionary to fill with summary statistics for the current object
        self.file_data_summary = {}
        if file_name:
            self.file_data_summary['File Name'] = file_name
        if group_name:
            self.file_data_summary['Group Name'] = group_name
        self.file_data_summary['Num Boxes'] = self.num_boxes
        if len(self.acf_results) > 0:
            self.file_data_summary['Ch1 % Zero Boxes'] = ch1_pcnt_no_period
            self.file_data_summary['Ch1 Mean Period'] = np.mean(ch1_period_measurements[5:])
            self.file_data_summary['Ch1 Median Period'] = np.median(ch1_period_measurements[5:])
            self.file_data_summary['Ch1 StdDev Period'] = np.std(ch1_period_measurements[5:])
            self.file_data_summary['Ch1 SEM Period'] = np.std(ch1_period_measurements[5:]) / np.sqrt(len(ch1_period_measurements[5:]))

            if self.num_channels == 2:
                self.file_data_summary['Ch2 % Zero Boxes'] = ch2_pcnt_no_period
                self.file_data_summary['Ch2 Mean Period'] = np.mean(ch2_period_measurements[5:])
                self.file_data_summary['Ch2 Median Period'] = np.median(ch2_period_measurements[5:])
                self.file_data_summary['Ch2 StdDev Period'] = np.std(ch2_period_measurements[5:])
                self.file_data_summary['Ch2 SEM Period'] = np.std(ch2_period_measurements[5:]) / np.sqrt(len(ch2_period_measurements[5:]))

        if len(self.ccf_results) > 0:
            self.file_data_summary['Mean Shift'] = np.mean(shift_measurements[5:])
            self.file_data_summary['Median Shift'] = np.median(shift_measurements[5:])
            self.file_data_summary['StdDev Shift'] = np.std(shift_measurements[5:])
            self.file_data_summary['SEM Shift'] = np.std(shift_measurements[5:]) / np.sqrt(len(shift_measurements[5:]))

        if len(self.peak_results) > 0:
            self.file_data_summary['Ch1 Mean Width'] = np.mean(ch1_width_measurements[5:])
            self.file_data_summary['Ch1 Median Width'] = np.median(ch1_width_measurements[5:])
            self.file_data_summary['Ch1 StdDev Width'] = np.std(ch1_width_measurements[5:])
            self.file_data_summary['Ch1 SEM Width'] = np.std(ch1_width_measurements[5:]) / np.sqrt(len(ch1_width_measurements[5:]))
            self.file_data_summary['Ch1 Mean Max'] = np.mean(ch1_max_measurements[5:])
            self.file_data_summary['Ch1 Median Max'] = np.median(ch1_max_measurements[5:])
            self.file_data_summary['Ch1 StdDev Max'] = np.std(ch1_max_measurements[5:])
            self.file_data_summary['Ch1 SEM Max'] = np.std(ch1_max_measurements[5:]) / np.sqrt(len(ch1_max_measurements[5:]))
            self.file_data_summary['Ch1 Mean Min'] = np.mean(ch1_min_measurements[5:])
            self.file_data_summary['Ch1 Median Min'] = np.median(ch1_min_measurements[5:])
            self.file_data_summary['Ch1 StdDev Min'] = np.std(ch1_min_measurements[5:])
            self.file_data_summary['Ch1 SEM Min'] = np.std(ch1_min_measurements[5:]) / np.sqrt(len(ch1_min_measurements[5:]))
            self.file_data_summary['Ch1 Mean Amp'] = np.mean(ch1_amp_measurements[5:])
            self.file_data_summary['Ch1 Median Amp'] = np.median(ch1_amp_measurements[5:])
            self.file_data_summary['Ch1 StdDev Amp'] = np.std(ch1_amp_measurements[5:])
            self.file_data_summary['Ch1 SEM Amp'] = np.std(ch1_amp_measurements[5:]) / np.sqrt(len(ch1_amp_measurements[5:]))
            self.file_data_summary['Ch1 Mean RelAmp'] = np.mean(ch1_relAmp_measurements[5:])
            self.file_data_summary['Ch1 Median RelAmp'] = np.median(ch1_relAmp_measurements[5:])
            self.file_data_summary['Ch1 StdDev RelAmp'] = np.std(ch1_relAmp_measurements[5:])
            self.file_data_summary['Ch1 SEM RelAmp'] = np.std(ch1_relAmp_measurements[5:]) / np.sqrt(len(ch1_relAmp_measurements[5:]))
            
            if self.num_channels == 2:
                self.file_data_summary['Ch2 Mean Width'] = np.mean(ch2_width_measurements[5:])
                self.file_data_summary['Ch2 Median Width'] = np.median(ch2_width_measurements[5:])
                self.file_data_summary['Ch2 StdDev Width'] = np.std(ch2_width_measurements[5:])
                self.file_data_summary['Ch2 SEM Width'] = np.std(ch2_width_measurements[5:]) / np.sqrt(len(ch2_width_measurements[5:]))
                self.file_data_summary['Ch2 Mean Max'] = np.mean(ch2_max_measurements[5:])
                self.file_data_summary['Ch2 Median Max'] = np.median(ch2_max_measurements[5:])
                self.file_data_summary['Ch2 StdDev Max'] = np.std(ch2_max_measurements[5:])
                self.file_data_summary['Ch2 SEM Max'] = np.std(ch2_max_measurements[5:]) / np.sqrt(len(ch2_max_measurements[5:]))
                self.file_data_summary['Ch2 Mean Min'] = np.mean(ch2_min_measurements[5:])
                self.file_data_summary['Ch2 Median Min'] = np.median(ch2_min_measurements[5:])
                self.file_data_summary['Ch2 StdDev Min'] = np.std(ch2_min_measurements[5:])
                self.file_data_summary['Ch2 SEM Min'] = np.std(ch2_min_measurements[5:]) / np.sqrt(len(ch2_min_measurements[5:]))
                self.file_data_summary['Ch2 Mean Amp'] = np.mean(ch2_amp_measurements[5:])
                self.file_data_summary['Ch2 Median Amp'] = np.median(ch2_amp_measurements[5:])
                self.file_data_summary['Ch2 StdDev Amp'] = np.std(ch2_amp_measurements[5:])
                self.file_data_summary['Ch2 SEM Amp'] = np.std(ch2_amp_measurements[5:]) / np.sqrt(len(ch2_amp_measurements[5:]))
                self.file_data_summary['Ch2 Mean RelAmp'] = np.mean(ch2_relAmp_measurements[5:])
                self.file_data_summary['Ch2 Median RelAmp'] = np.median(ch2_relAmp_measurements[5:])
                self.file_data_summary['Ch2 StdDev RelAmp'] = np.std(ch2_relAmp_measurements[5:])
                self.file_data_summary['Ch2 SEM RelAmp'] = np.std(ch2_relAmp_measurements[5:]) / np.sqrt(len(ch2_relAmp_measurements[5:]))

        return self.im_measurements, self.file_data_summary

    # function to plot a summary of the period measurements
    def plot_mean_CF(self):
        
        def return_figure(self, num_points: int, arr: np.ndarray, shifts_or_periods: np.ndarray, channel: str, type_of_plot: str, type_of_measurement: str):
            
            fig, ax = plt.subplot_mosaic(mosaic = '''AA
                                                     BC''')
            arr_mean = np.mean(arr, axis = 0)
            arr_std = np.std(arr, axis = 0)
            ax['A'].plot(arr_mean, color='blue')
            ax['A'].fill_between(np.arange(num_points), 
                                            arr_mean - arr_std, 
                                            arr_mean + arr_std, 
                                            color='blue', 
                                            alpha=0.2)
            ax['A'].set_title(f'{channel} Mean {type_of_plot} Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            ax['B'].set_xlabel(f'Histogram of {type_of_measurement} values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of {type_of_measurement} values')
            ax['C'].set_ylabel(f'Measured {type_of_measurement} (frames)')
            fig.subplots_adjust(hspace=0.25, wspace=0.5)   
            return fig

        # num points on x-axis
        x_axis_points = self.num_frames*2 - 1
        # empty dict to fill with figures, in the event that we make more than one
        self.acf_figs = {}
        
        # populate the ACF data from each box into a single array
        if len(self.acf_results) > 0:
            ch1_acfs = np.zeros(shape=(self.num_boxes, x_axis_points))
            ch1_box_periods = np.zeros(shape=(self.num_boxes))
            for box in range(self.num_boxes):
                ch1_acfs[box] = self.acf_results[f'Ch1_ACF_box{box}'][1]
                ch1_box_periods[box] = self.acf_results[f'Ch1_ACF_box{box}'][0]
            # if channel 2 exists, do the same for it
            if self.num_channels == 2:
                ch2_acfs = np.zeros(shape=(self.num_boxes, x_axis_points))
                ch2_box_periods = np.zeros(shape=(self.num_boxes))
                for box in range(self.num_boxes):
                    ch2_acfs[box] = self.acf_results[f'Ch2_ACF_box{box}'][1]
                    ch2_box_periods[box] = self.acf_results[f'Ch2_ACF_box{box}'][0]
                
                fig2 = return_figure(self, x_axis_points, ch2_acfs, ch2_box_periods, 'Ch2', 'Autocorrelation', 'period')
                self.acf_figs['Ch2 ACF'] = fig2
                plt.close()
        
        fig1 = return_figure(self, x_axis_points, ch1_acfs, ch1_box_periods, 'Ch1', 'Autocorrelation', 'period')
        self.acf_figs['Ch1 ACF'] = fig1
        plt.close()

        if len(self.ccf_results) > 0:
            mean_ccfs = np.zeros(shape=(self.num_boxes, x_axis_points))
            box_shifts = np.zeros(shape=(self.num_boxes))
            for box in range(self.num_boxes):
                mean_ccfs[box] = self.ccf_results[f'CCF_box{box}'][1]
                box_shifts[box] = self.ccf_results[f'CCF_box{box}'][0]
            
            fig3 = return_figure(self, x_axis_points, mean_ccfs, box_shifts, 'Mean', 'Cross-correlation', 'shift')
            self.acf_figs['Mean CCF'] = fig3
            plt.close()
        
        return self.acf_figs


        


