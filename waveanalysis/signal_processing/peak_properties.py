import numpy as np
import scipy.signal as sig

def calc_indv_peak_props(
        signal:np.ndarray
) -> tuple:
    peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

    # If peaks detected, calculate properties, otherwise return NaNs
    if len(peaks) > 0:
        widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
        proms, _, _ = sig.peak_prominences(signal, peaks)
        mean_width = np.mean(widths, axis=0)
        mean_max = np.mean(signal[peaks], axis = 0)
        mean_min = np.mean(signal[peaks]-proms, axis = 0)

        # calculate the left and right bases of the peaks, then peak offsets
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
    
    return mean_width, mean_max, mean_min, mean_offset, peaks, proms, heights, leftIndex, rightIndex, midpoints, peak_offsets, left_bases, right_bases
