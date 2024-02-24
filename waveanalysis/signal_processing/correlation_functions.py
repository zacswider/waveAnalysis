
import numpy as np
from scipy import signal as sig


def create_acf_curves_calc_period(
    signal: np.ndarray,
    num_frames_or_rollsize: int,
    peak_thresh: float,
) -> tuple[float, np.ndarray]:
    """
    This function normalizes the input signal and computes the autocorrelation curve.
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