import numpy as np
import scipy.signal as sig

def calc_indv_peak_props(
        signal:np.ndarray
) -> tuple:
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
    
    return mean_width, mean_max, mean_min, peaks, proms, heights, leftIndex, rightIndex

def calc_indv_peak_offset(
    num_channels:int,
    num_bins:int,
    bin_values:np.ndarray,
    analysis_type:str
) -> dict:
    indv_peak_offsets = {}

    # Loop through channels and bins for standard or kymograph analysis
    for channel in range(num_channels):
        for bin in range(num_bins):
            if analysis_type == "standard":
                signal = sig.savgol_filter(bin_values[:,channel, bin], window_length = 11, polyorder = 2)  
            else:                     
                signal = sig.savgol_filter(bin_values[channel, bin], window_length = 11, polyorder = 2)   
            
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

            # If peaks detected, calculate properties, otherwise return NaNs
            if len(peaks) > 0:
                _, left_base, right_base = sig.peak_prominences(signal, peaks)

                midpoint = (left_base + right_base) / 2
                peak_offsets = peaks - midpoint
                indv_peak_offsets[f'Ch {channel} Bin {bin}'] ={'offsets': peak_offsets, 
                                                                'midpoints': midpoint,
                                                                'left_base': left_base, 
                                                                'right_base': right_base
                                                                }

            else:
                indv_peak_offsets[f'Ch {channel} Bin {bin}'] = np.nan

    return indv_peak_offsets