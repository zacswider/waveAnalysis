import numpy as np
from scipy import signal as sig

def calc_indv_ACF_workflow(
    bin_values: np.ndarray,
    img_props: dict
) -> np.ndarray: 
    
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    num_frames = img_props['num_frames']
    acf_peak_thresh = img_props['peak_thresh']
    analysis_type = img_props['analysis_type']

    indv_acfs = np.zeros(shape=(num_channels, num_bins, num_frames * 2 - 1))

    for channel in range(num_channels):
        for bin in range(num_bins):
            signal = bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
            acf_curve = calc_indv_ACF(signal=signal, num_frames=num_frames, peak_thresh=acf_peak_thresh)
            indv_acfs[channel, bin] = acf_curve

    return indv_acfs

def calc_indv_ACF(
    signal: np.ndarray,
    num_frames: int,
    peak_thresh: float,
) -> (np.ndarray, np.ndarray): #type: ignore

    # calc autocorrelation and normalize
    corr_signal = signal - np.mean(signal)
    acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
    acf_curve = acf_curve / (num_frames * np.std(signal) ** 2)

    peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
    if len(peaks) < 2:
        acf_curve = np.full((num_frames * 2 - 1), np.nan)

    return acf_curve

def calc_indv_period_workflow(
    acf_curve: np.ndarray,
    img_props: dict
) -> np.ndarray: 
    
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    acf_peak_thresh = img_props['peak_thresh']

    indv_periods = np.zeros(shape=(num_channels, num_bins))

    for channel in range(num_channels):
        for bin in range(num_bins):
            period = calc_indv_period(acf_curve=acf_curve[channel, bin], peak_thresh=acf_peak_thresh)
            indv_periods[channel, bin] = period

    return indv_periods

def calc_indv_period(
    acf_curve: np.ndarray,
    peak_thresh: float,
) -> float:
    # Find peaks in the autocorrelation curve, Calculate absolute differences between peaks and center
    peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
    peaks_abs = np.abs(peaks - acf_curve.shape[0] // 2)

    # If peaks are identified, pick the closest one to the center as the period
    period = np.min(peaks_abs[np.nonzero(peaks_abs)]) if len(peaks) > 1 else np.nan
        
    return period

def calc_indv_CCF_workflow(
    bin_values: np.ndarray,
    img_props: dict
) -> np.ndarray:
    
    num_combos = img_props['num_combos']
    num_bins = img_props['num_bins']
    num_frames = img_props['num_frames']
    channel_combos = img_props['channel_combos']
    analysis_type = img_props['analysis_type']

    indv_ccfs = np.zeros(shape=(num_combos, num_bins, num_frames*2-1))
    
    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):
            if analysis_type == 'standard':
                signal1 = sig.savgol_filter(bin_values[:, combo[0], bin], window_length=11, polyorder=3)
                signal2 = sig.savgol_filter(bin_values[:, combo[1], bin], window_length=11, polyorder=3)
            else:
                signal1 = sig.savgol_filter(bin_values[combo[0], bin], window_length=11, polyorder=3)
                signal2 = sig.savgol_filter(bin_values[combo[1], bin], window_length=11, polyorder=3)
            ccf = calc_indv_CCF(signal1=signal1, signal2=signal2, num_frames=num_frames)
            indv_ccfs[combo_number, bin] = ccf

    return indv_ccfs

def calc_indv_CCF(
    signal1: np.ndarray,
    signal2: np.ndarray,
    num_frames: int,
) -> (np.ndarray, np.ndarray): #type: ignore
    """
    This method computes the cross-correlation functions (CCFs) for each combination of channels.
    It also identifies peaks in the CCF curves to estimate shifts.

    Returns:
        - indv_shifts (numpy.ndarray): Array of shifts between signals.
        - indv_ccfs (numpy.ndarray): Array of cross-correlation functions.
        - channel_combos (list): List of channel combinations.
    """

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

        if len(peaks) < 2:
            cc_curve = np.full((num_frames * 2 - 1), np.nan)

    else:
        # If no peaks found, return NaNs
        cc_curve = np.full((num_frames * 2 - 1), np.nan)

    return cc_curve

def calc_indv_shift_workflow(
    indv_ccfs: np.ndarray,
    indv_periods: np.ndarray,
    img_props: dict
) -> np.ndarray:
    
    num_combos = img_props['num_combos']
    num_bins = img_props['num_bins']
    channel_combos = img_props['channel_combos']
    
    indv_shifts = np.zeros(shape=(num_combos, num_bins))

    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):
            shift = calc_indv_shift(cc_curve=indv_ccfs[combo_number, bin])
            average_period = np.mean(indv_periods[:, bin]) # If the shift is too small, correct it
            shift = small_shifts_correction(delay_frames=shift, average_period=average_period)
            indv_shifts[combo_number, bin] = shift

    return indv_shifts

def calc_indv_shift(cc_curve: np.ndarray) -> np.ndarray:
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
    
    return delay_frames

def small_shifts_correction(delay_frames, average_period):
    if abs(delay_frames) > abs(average_period * .6):
        if delay_frames < 0:
            delay_frames = delay_frames + average_period
        elif delay_frames > 0:
            delay_frames = delay_frames - average_period

    return delay_frames
