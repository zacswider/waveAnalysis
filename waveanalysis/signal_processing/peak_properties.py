import numpy as np
from tqdm import tqdm
import scipy.signal as sig

def indv_props_standard_kymo(
    num_channels: int,
    total_bins: int,
    analysis_type: str,
    bin_values: np.ndarray
): # type: ignore

    ind_peak_widths = np.zeros(shape=(num_channels, total_bins))
    ind_peak_maxs = np.zeros(shape=(num_channels, total_bins))
    ind_peak_mins = np.zeros(shape=(num_channels, total_bins))
    ind_peak_props = {}

    for channel in range(num_channels):
        for bin in range(total_bins):
            if analysis_type == "standard":
                signal = sig.savgol_filter(bin_values[:,channel, bin], window_length = 11, polyorder = 2)   
            else:                     
                signal = sig.savgol_filter(bin_values[channel, bin], window_length = 11, polyorder = 2)  

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

            # Store peak measurements for each bin in each channel
            ind_peak_widths[channel, bin] = mean_width
            ind_peak_maxs[channel, bin] = mean_max
            ind_peak_mins[channel, bin] = mean_min
            ind_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': signal, 
                                                    'peaks': peaks,
                                                    'proms': proms, 
                                                    'heights': heights, 
                                                    'leftIndex': leftIndex, 
                                                    'rightIndex': rightIndex}
            
     # Calculate additional peak properties
    ind_peak_amps = ind_peak_maxs - ind_peak_mins
    ind_peak_rel_amps = ind_peak_amps / ind_peak_mins
    
    return ind_peak_widths, ind_peak_maxs, ind_peak_mins, ind_peak_amps, ind_peak_rel_amps, ind_peak_props
    

def indv_props_rolling(
    num_submovies: int,
    num_channels: int,
    total_bins: int,
    xpix: int,
    ypix: int,
    roll_size: int,
    roll_by: int,
    bin_values: np.ndarray
): # type: ignore
    
    ind_peak_widths = np.zeros(shape=(num_submovies, num_channels, total_bins))
    ind_peak_maxs = np.zeros(shape=(num_submovies, num_channels, total_bins))
    ind_peak_mins = np.zeros(shape=(num_submovies, num_channels, total_bins))
    
    its = num_submovies*num_channels*xpix*ypix
    with tqdm(total = its, miniters=its/100) as pbar:
        pbar.set_description('Rolling Peak Prop Calculations: ')
        for submovie in range(num_submovies):
            for channel in range(num_channels):
                for bin in range(total_bins):
                    pbar.update(1)
                    signal = sig.savgol_filter(bin_values[roll_by*submovie : roll_size + roll_by*submovie, channel, bin], window_length=11, polyorder=2)

                    peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

                    # If peaks detected, calculate properties, otherwise return NaNs
                    if len(peaks) > 0:
                        proms, _, _ = sig.peak_prominences(signal, peaks)
                        widths = sig.peak_widths(signal, peaks, rel_height=0.5)
                        mean_width = np.mean(widths, axis=0)                        
                        mean_max = np.mean(signal[peaks], axis = 0)
                        mean_min = np.mean(signal[peaks]-proms, axis = 0)

                        if len(peaks) > 0:
                            mean_width = np.mean(widths, axis=0)                        
                            mean_max = np.mean(signal[peaks], axis = 0)
                            mean_min = np.mean(signal[peaks]-proms, axis = 0)
                    else:
                        mean_width = np.nan
                        mean_max = np.nan
                        mean_min = np.nan
                        peaks = np.nan
                        proms = np.nan 

                    # Store peak measurements for each bin in each channel of a submovie
                    ind_peak_widths[submovie, channel, bin] = mean_width
                    ind_peak_maxs[submovie, channel, bin] = mean_max
                    ind_peak_mins[submovie, channel, bin] = mean_min
    
    ind_peak_amps = ind_peak_maxs - ind_peak_mins
    ind_peak_rel_amps = ind_peak_amps / ind_peak_mins

    return ind_peak_widths, ind_peak_maxs, ind_peak_mins, ind_peak_amps, ind_peak_rel_amps
