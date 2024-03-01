from tqdm import tqdm
import numpy as np
from scipy import signal as sig

# TODO: combine the rolling, standard, and kymograph analysis into one function
# TODO: likely will need to make separate functions for the different types of analysis

def calc_indv_standard_kymo_ACFs_periods(
                           num_channels: int,
                           num_bins: int,
                           num_frames: int,
                           bin_values: np.ndarray,
                           analysis_type: str,
                           peak_thresh: float = 0.1
) -> (np.ndarray, np.ndarray):
    # Initialize arrays to store period measurements and autocorrelation curves
    periods = np.zeros(shape=(num_channels, num_bins))
    acfs = np.zeros(shape=(num_channels, num_bins, num_frames * 2 - 1))

# Loop through channels and bins for standard or kymograph analysis
    for channel in range(num_channels):
        for bin in range(num_bins):
            signal = bin_values[:, channel, bin] if analysis_type == "standard" else bin_values[channel, bin]

            corr_signal = signal - np.mean(signal)
            acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
            # Normalize the autocorrelation curve
            acf_curve = acf_curve / (num_frames * np.std(signal) ** 2)
            # Find peaks in the autocorrelation curve
            peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
            # Calculate absolute differences between peaks and center
            peaks_abs = np.abs(peaks - acf_curve.shape[0] // 2)
            # If peaks are identified, pick the closest one to the center
            if len(peaks) > 1:
                delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
            else:
                # Otherwise, return NaNs for both delay and autocorrelation curve
                delay = np.nan
                acf_curve = np.full((num_frames * 2 - 1), np.nan)

            periods[channel, bin] = delay
            acfs[channel, bin] = acf_curve
                        
    return acfs, periods

def calc_indv_rolling_ACFs_periods(
                           num_channels: int,
                           num_bins: int,
                           bin_values: np.ndarray,
                           roll_size: int,
                           roll_by: int,
                           num_submovies: int,
                           num_x_bins: int,
                           num_y_bins: int,
                           peak_thresh: float = 0.1
):
    """
    This method computes the autocorrelation functions (ACFs) for each channel and bin of the analyzed data.
    It also identifies peaks in the ACF curves to estimate periods.

    Parameters:
        - peak_thresh (float): Threshold for peak detection in the ACF curves. Defaults to 0.1.

    Returns:
        - acfs (numpy.ndarray): Array of autocorrelation functions.
        - periods (numpy.ndarray): Array of periods estimated from the ACF peaks.
    """
    
    # Initialize arrays to store period measurements and autocorrelation curves

    periods = np.zeros(shape=(num_submovies, num_channels, num_bins))
    acfs = np.zeros(shape=(num_submovies, num_channels, num_bins, roll_size * 2 - 1))
    # Loop through submovies, channels, and bins
    its = num_submovies*num_channels*num_x_bins*num_y_bins
    with tqdm(total = its, miniters=its/100) as pbar:
        pbar.set_description( 'Periods: ')
        for submovie in range(num_submovies):
            for channel in range(num_channels):
                for bin in range(num_bins):
                    pbar.update(1)
                    # Extract signal for rolling autocorrelation calculation
                    signal = bin_values[roll_by * submovie: roll_size + roll_by * submovie, channel, bin]

                    corr_signal = signal - np.mean(signal)
                    acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
                    # Normalize the autocorrelation curve
                    acf_curve = acf_curve / (roll_size * np.std(signal) ** 2)
                    # Find peaks in the autocorrelation curve
                    peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
                    # Calculate absolute differences between peaks and center
                    peaks_abs = np.abs(peaks - acf_curve.shape[0] // 2)
                    # If peaks are identified, pick the closest one to the center
                    if len(peaks) > 1:
                        delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
                    else:
                        # Otherwise, return NaNs for both delay and autocorrelation curve
                        delay = np.nan
                        acf_curve = np.full((roll_size * 2 - 1), np.nan)

                    periods[submovie, channel, bin] = delay
                    acfs[submovie, channel, bin] = acf_curve
                    
    return acfs, periods


def calc_indv_CCFs_shifts_channelCombos(
    num_channels: int,
    num_bins: int,
    num_frames: int,
    bin_values: np.ndarray,
    analysis_type: str,
    roll_size: int = np.nan,
    roll_by: int = np.nan,
    num_submovies: int = np.nan,
    periods: np.ndarray = np.nan
 ):
    """
    This method computes the cross-correlation functions (CCFs) for each combination of channels.
    It also identifies peaks in the CCF curves to estimate shifts.

    Returns:
        - indv_shifts (numpy.ndarray): Array of shifts between signals.
        - indv_ccfs (numpy.ndarray): Array of cross-correlation functions.
        - channel_combos (list): List of channel combinations.
    """
    
    # Initialize arrays to store shifts and cross-correlation curves
    channels = list(range(num_channels))
    channel_combos = []
    for i in range(num_channels):
        for j in channels[i+1:]:
            channel_combos.append([channels[i],j])
    num_combos = len(channel_combos)

    # Initialize arrays to store shifts and cross-correlation curves
    indv_shifts = np.zeros(shape=(num_combos, num_bins))
    indv_ccfs = np.zeros(shape=(num_combos, num_bins, num_frames*2-1))

    # Loop through combos for standard or kymograph analysis
    if analysis_type != "rolling":
        for combo_number, combo in enumerate(channel_combos):
            for bin in range(num_bins):
                if analysis_type == "standard":
                    signal1 = bin_values[:, combo[0], bin]
                    signal2 = bin_values[:, combo[1], bin]
                else:
                    signal1 = bin_values[combo[0], bin]
                    signal2 = bin_values[combo[1], bin]

                signal1 = sig.savgol_filter(signal1, window_length=11, polyorder=3)
                signal2 = sig.savgol_filter(signal2, window_length=11, polyorder=3)
                peaks1, _ = sig.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
                peaks2, _ = sig.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

                # If peaks are found in both signals
                if len(peaks1) > 0 and len(peaks2) > 0:
                    corr_signal1 = signal1 - signal1.mean()
                    corr_signal2 = signal2 - signal2.mean()
                    # Calculate cross-correlation curve
                    cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')

                    cc_curve = sig.savgol_filter(cc_curve, window_length=11, polyorder=3)
                    cc_curve = cc_curve / (num_frames * signal1.std() * signal2.std())
                    # Find peaks in the cross-correlation curve
                    peaks, _ = sig.find_peaks(cc_curve, prominence=0.1)
                    peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
                    # If multiple peaks found, select the one closest to the center
                    if len(peaks) > 1:
                        delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                        delayIndex = peaks[delay]
                        delay_frames = delayIndex - cc_curve.shape[0] // 2
                    # Otherwise, return NaNs
                    else:
                        delay_frames = np.nan
                        cc_curve = np.full((num_frames * 2 - 1), np.nan)
                else:
                    # If no peaks found, return NaNs
                    delay_frames = np.nan
                    cc_curve = np.full((num_frames * 2 - 1), np.nan)
    
                # The script has issues when the shift is very small or none, so minus the average period from the two channels
                average_period = np.mean(periods[:, bin])
                if abs(delay_frames) > abs(average_period * .6):
                    if delay_frames < 0:
                        delay_frames = delay_frames + average_period
                    elif delay_frames > 0:
                        delay_frames = delay_frames - average_period

                indv_shifts[combo_number, bin] = delay_frames
                indv_ccfs[combo_number, bin] = cc_curve
    

    # If rolling analysis
    elif analysis_type == "rolling":
        # Initialize arrays to store shifts and cross-correlation curves
        indv_shifts = np.zeros(shape=(num_submovies, num_combos, num_bins))
        indv_ccfs = np.zeros(shape=(num_submovies, num_combos, num_bins, roll_size*2-1))
        its = num_submovies*num_combos*num_bins
        with tqdm(total = its, miniters=its/100) as pbar:
            pbar.set_description( 'Shifts: ')
            for submovie in range(num_submovies):
                for combo_number, combo in enumerate(channel_combos):
                    for bin in range(num_bins):
                        pbar.update(1)
                        signal1 = bin_values[roll_by*submovie : roll_size + roll_by*submovie, combo[0], bin]
                        signal2 = bin_values[roll_by*submovie : roll_size + roll_by*submovie, combo[1], bin]

                                    # Smoothing signals and finding peaks
                        signal1 = sig.savgol_filter(signal1, window_length=11, polyorder=3)
                        signal2 = sig.savgol_filter(signal2, window_length=11, polyorder=3)
                        peaks1, _ = sig.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
                        peaks2, _ = sig.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

                        # If peaks are found in both signals
                        if len(peaks1) > 0 and len(peaks2) > 0:
                            corr_signal1 = signal1 - signal1.mean()
                            corr_signal2 = signal2 - signal2.mean()
                            # Calculate cross-correlation curve
                            cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
                            
                            cc_curve = cc_curve / (roll_size * signal1.std() * signal2.std())
                            
                            # Find peaks in the cross-correlation curve
                            peaks, _ = sig.find_peaks(cc_curve, prominence=0.1)
                            peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
                            # If multiple peaks found, select the one closest to the center
                            if len(peaks) > 1:
                                delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                                delayIndex = peaks[delay]
                                delay_frames = delayIndex - cc_curve.shape[0] // 2
                            # Otherwise, return NaNs
                            else:
                                delay_frames = np.nan
                                cc_curve = np.full((roll_size*2-1), np.nan)
                        else:
                            # If no peaks found, return NaNs
                            delay_frames = np.nan
                            cc_curve = np.full((roll_size*2-1), np.nan)

                        indv_shifts[submovie, combo_number, bin] = delay_frames
                        indv_ccfs[submovie, combo_number, bin] = cc_curve

    return indv_shifts, indv_ccfs, channel_combos