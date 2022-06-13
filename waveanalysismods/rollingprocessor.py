import numpy as np
import pandas as pd
import scipy.signal as sig
from tqdm import tqdm
import matplotlib.pyplot as plt
from tifffile import imread, imwrite, TiffFile
import scipy.ndimage as nd
np.seterr(divide='ignore', invalid='ignore')

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

