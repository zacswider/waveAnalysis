
import numpy as np
from scipy import signal as sig


def acf_shifts(
    signal: np.ndarray,
    num_frames_or_rollsize: int,
    peak_thresh: float,
) -> tuple[float, np.ndarray]:
    """
    This function normalizes the input signal and computes the aupyttocorrelation curve.
    It identifies peaks in the autocorrelation curve to estimate the delay.

    Parameters:
        - signal (numpy.ndarray): Input signal.
        - num_frames_or_rows_or_rollsize (int): Number of frames or roll size for normalization.

    Returns:
        - delay (float): Delay estimated from the autocorrelation curve.
        - acf_curve (numpy.ndarray): Autocorrelation curve of the normalized signal.
    """
    corr_signal = signal - np.mean(signal)
    acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
    # Normalize the autocorrelation curve
    acf_curve = acf_curve / (num_frames_or_rollsize * np.std(signal) ** 2)
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
        acf_curve = np.full((num_frames_or_rollsize * 2 - 1), np.nan)
    return delay, acf_curve

def calc_shifts_CCF_curves(
    signal1: np.array, 
    signal2: np.array, 
    prominence: float = 0.1, 
    num_frames: int = np.nan,
    roll_size: int = np.nan,
) -> tuple[float, np.ndarray]:
   
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
        if roll_size:
            cc_curve = cc_curve / (roll_size * signal1.std() * signal2.std())
        else:
            cc_curve = sig.savgol_filter(cc_curve, window_length=11, polyorder=3)
            cc_curve = cc_curve / (num_frames * signal1.std() * signal2.std())
        # Find peaks in the cross-correlation curve
        peaks, _ = sig.find_peaks(cc_curve, prominence=prominence)
        peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
        # If multiple peaks found, select the one closest to the center
        if len(peaks) > 1:
            delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
            delayIndex = peaks[delay]
            delay_frames = delayIndex - cc_curve.shape[0] // 2
        # Otherwise, return NaNs
        else:
            delay_frames = np.nan
            cc_curve = np.full((roll_size*2-1 if roll_size else num_frames * 2 - 1), np.nan)
    else:
        # If no peaks found, return NaNs
        delay_frames = np.nan
        cc_curve = np.full((roll_size*2-1 if roll_size else num_frames * 2 - 1), np.nan)

    return delay_frames, cc_curve