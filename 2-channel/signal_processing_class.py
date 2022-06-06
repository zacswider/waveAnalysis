import numpy as np
import scipy.signal as sig 
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
        print(f'number of channels is {self.num_channels}')
        print(f'number of slices is {self.num_slices}')
        print(f'number of frames is {self.num_frames}')
        self.image = self.image.reshape(self.num_frames, 
                                        self.num_slices, 
                                        self.num_channels, 
                                        self.image.shape[-2], 
                                        self.image.shape[-1])
        print(f'Image dimensions after reshaping {self.image.shape}')
        # max project image stack if num_slices > 1
        if self.num_slices > 1:
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

        # return the time-axis means for each channel
        self.box_means = np.zeros((self.x_boxes, self.y_boxes, self.num_channels, self.num_frames))
        for channel in range(self.num_channels):
            for x in range(self.x_boxes):
                for y in range(self.y_boxes):
                    self.box_means[x, y, channel] = np.mean(self.image[:, 0, channel, x*self.box_size:(x+1)*self.box_size, y*self.box_size:(y+1)*self.box_size], axis=(1,2))
        # reshape into 2D array. Shape is (channels, boxes, frames)
        self.box_means = self.box_means.reshape((self.num_channels, self.x_boxes*self.y_boxes, self.num_frames))

    # function to return the autocorrelation of each box in the image stack for each channel
    def calc_ACF(self, peak_thresh):
        '''
        Returns a dictionary containing the channel identify and box number as keys and the
        calculated period and autocorrelation curve as a values in a tuple.
        '''
        # empty dictionary to fill with autocorrelation values
        self.acf_results = {}
        for channel in range(self.num_channels):
            for box_num in range(self.x_boxes*self.y_boxes):
                # calculate full autocorrelation
                signal = self.box_means[channel, box_num]
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
        self.ccf_results = {}
        assert self.num_channels == 2, 'CCF only works for 2 channels'
        for box_num in range(self.x_boxes*self.y_boxes):
            # calculate full cross-correlation
            signal_1 = self.box_means[0, box_num]
            signal_2 = self.box_means[1, box_num]
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
        # empty dictionary to fill with peak properties
        self.peak_results = {}
        for channel in range(self.num_channels):
            for box_num in range(self.x_boxes*self.y_boxes):
                signal = sig.savgol_filter(self.box_means[channel, box_num], window_length = 11, polyorder = 2)
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
