from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.signal as sig

def calc_indv_peak_props_standard_kymo(
    num_channels:int,
    num_bins:int,
    bin_values:np.ndarray,
    analysis_type:str
):
    """
    This method computes various peak properties for each channel and bin of the analyzed data.

    Returns:
        - indv_peak_widths (numpy.ndarray): Array of peak widths.
        - indv_peak_maxs (numpy.ndarray): Array of peak maximum values.
        - indv_peak_mins (numpy.ndarray): Array of peak minimum values.
        - indv_peak_amps (numpy.ndarray): Array of peak amplitudes.
        - indv_peak_rel_amps (numpy.ndarray): Array of relative peak amplitudes.
        - indv_peak_props (dict): Dictionary containing additional peak properties.
    """

    # Initialize arrays/dictionary to store peak measurements
    indv_peak_widths = np.zeros(shape=(num_channels, num_bins))
    indv_peak_maxs = np.zeros(shape=(num_channels, num_bins))
    indv_peak_mins = np.zeros(shape=(num_channels, num_bins))
    indv_peak_props = {}

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
            indv_peak_widths[channel, bin] = mean_width
            indv_peak_maxs[channel, bin] = mean_max
            indv_peak_mins[channel, bin] = mean_min
            indv_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': signal, 
                                                    'peaks': peaks,
                                                    'proms': proms, 
                                                    'heights': heights, 
                                                    'leftIndex': leftIndex, 
                                                    'rightIndex': rightIndex}

    # Calculate additional peak properties
    indv_peak_amps = indv_peak_maxs - indv_peak_mins
    indv_peak_rel_amps = indv_peak_amps / indv_peak_mins

    
    return indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_amps, indv_peak_rel_amps, indv_peak_props

def calc_indv_peak_props_rolling(
    num_channels:int,
    num_bins:int,
    bin_values:np.ndarray,
    num_submovies:int = None,
    roll_by:int = None,
    roll_size:int = None,
    num_x_bins:int = None,
    num_y_bins:int = None
):
    """
    This method computes various peak properties for each channel and bin of the analyzed data.

    Returns:
        - indv_peak_widths (numpy.ndarray): Array of peak widths.
        - indv_peak_maxs (numpy.ndarray): Array of peak maximum values.
        - indv_peak_mins (numpy.ndarray): Array of peak minimum values.
        - indv_peak_amps (numpy.ndarray): Array of peak amplitudes.
        - indv_peak_rel_amps (numpy.ndarray): Array of relative peak amplitudes.
        - indv_peak_props (dict): Dictionary containing additional peak properties.
    """

    # Initialize arrays/dictionary to store peak measurements
    indv_peak_widths = np.zeros(shape=(num_submovies, num_channels, num_bins))
    indv_peak_maxs = np.zeros(shape=(num_submovies, num_channels, num_bins))
    indv_peak_mins = np.zeros(shape=(num_submovies, num_channels, num_bins))

    its = num_submovies*num_channels*num_x_bins*num_y_bins
    with tqdm(total = its, miniters=its/100) as pbar:
        pbar.set_description('Peak Props: ')
        for submovie in range(num_submovies):
            for channel in range(num_channels):
                for bin in range(num_bins):
                    pbar.update(1)
                    signal = sig.savgol_filter(bin_values[roll_by*submovie : roll_size + roll_by*submovie, channel, bin], window_length=11, polyorder=2)
                    
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

                    # Store peak measurements for each bin in each channel of a submovie
                    indv_peak_widths[submovie, channel, bin] = mean_width
                    indv_peak_maxs[submovie, channel, bin] = mean_max
                    indv_peak_mins[submovie, channel, bin] = mean_min
                    
    # Calculate additional peak properties
    indv_peak_amps = indv_peak_maxs - indv_peak_mins
    indv_peak_rel_amps = indv_peak_amps / indv_peak_mins

    
    return indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_amps, indv_peak_rel_amps

def calc_indv_peak_offset(
    num_channels:int,
    num_bins:int,
    bin_values:np.ndarray,
    analysis_type:str
):
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
            
    
    # Save indv_peak_offsets to a CSV file
    # df = pd.DataFrame(indv_peak_offsets)
    # df.columns = [f'Bin {i}' for i in range(1, num_bins+1)]
    # df.to_csv('/Users/domchom/Desktop/indv_peak_offsets.csv', index=False)

    return indv_peak_offsets