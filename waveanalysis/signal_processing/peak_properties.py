import numpy as np
import scipy.signal as sig

def calc_indv_peak_props_workflow(
    bin_values:np.ndarray,
    img_props:dict
) -> tuple:
    
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']

    indv_peak_widths = np.zeros(shape=(num_channels, num_bins))
    indv_peak_maxs = np.zeros(shape=(num_channels, num_bins))
    indv_peak_mins = np.zeros(shape=(num_channels, num_bins))
    indv_peak_offsets = np.zeros(shape=(num_channels, num_bins))
    indv_peak_props = {}

    for channel in range(num_channels):
        for bin in range(num_bins):
            signal = bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
            smoothed_signal = sig.savgol_filter(signal, window_length = 11, polyorder = 2)                 
            peaks, _ = sig.find_peaks(smoothed_signal, prominence=(np.max(signal)-np.min(signal))*0.1)

            # If peaks detected, calculate properties, otherwise return NaNs
            if len(peaks) > 0:
                widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
                proms, _, _ = sig.peak_prominences(signal, peaks)
                mean_width = np.mean(widths, axis=0)
                mean_max = np.mean(signal[peaks], axis = 0)
                mean_min = np.mean(signal[peaks]-proms, axis = 0)

                # calculate the left and right bases of the peaks, then midpoints and peak offsets
                _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=.99)
                midpoints = (leftIndex + rightIndex) / 2
                peak_offsets = peaks - midpoints
                # Check if one peak entirely encompasses another
                for i in range(len(peaks)):
                    for j in range(len(peaks)):
                        if i != j:  # Avoid self-comparison
                            if left_bases[j] >= left_bases[i] and right_bases[j] <= right_bases[i]:
                                # Peak j is entirely encompassed by peak i
                                left_bases[i] = np.nan
                                right_bases[i] = np.nan
                                peak_offsets[i] = np.nan
                                midpoints[i] = np.nan
                
                # Drop NaN values because it will mess up the mean calculation
                valid_indices = ~np.isnan(peak_offsets)
                valid_offsets = peak_offsets[valid_indices]
                # Calculate the mean of valid peak offsets
                mean_offset = np.nanmean(valid_offsets)
            else:
                mean_width = np.nan
                mean_max = np.nan
                mean_min = np.nan
                mean_offset = np.nan
                peaks = np.nan
                proms = np.nan 
                heights = np.nan
                leftIndex = np.nan
                rightIndex = np.nan

            indv_peak_widths[channel, bin] = mean_width
            indv_peak_maxs[channel, bin] = mean_max
            indv_peak_mins[channel, bin] = mean_min
            indv_peak_offsets[channel, bin] = mean_offset

            indv_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': smoothed_signal, 
                                                                'peaks': peaks,
                                                                'proms': proms, 
                                                                'heights': heights, 
                                                                'leftIndex': leftIndex, 
                                                                'rightIndex': rightIndex,
                                                                'midpoints': midpoints,
                                                                'peak_offsets': peak_offsets,
                                                                'left_base': left_bases,
                                                                'right_base': right_bases}
                        
                        # TODO: rename the keys to be more descriptive
    
    return indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_offsets, indv_peak_props
