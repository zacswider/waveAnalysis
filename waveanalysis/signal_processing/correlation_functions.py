from tqdm import tqdm
import numpy as np
from scipy import signal as sig

# TODO: combine both ACF functions into one, and also combine the rolling, standard, and kymograph analysis into one function

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

def calc_indv_ACFs_periods(num_channels: int,
                           total_bins: int,
                           num_frames: int,
                           bin_values: np.ndarray,
                           analysis_type: str,
                           roll_size: int,
                           roll_by: int,
                           num_submovies: int,
                           xpix: int,
                           ypix: int,
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
    periods = np.zeros(shape=(num_channels, total_bins))
    acfs = np.zeros(shape=(num_channels, total_bins, num_frames * 2 - 1))

    # Loop through channels and bins for standard or kymograph analysis
    if analysis_type != "rolling":
        for channel in range(num_channels):
            for bin in range(total_bins):
                signal = bin_values[:, channel, bin] if analysis_type == "standard" else bin_values[channel, bin, :]
                delay, acf_curve = create_acf_curves_calc_period(signal, num_frames_or_rollsize=num_frames, peak_thresh=peak_thresh)
                periods[channel, bin] = delay
                acfs[channel, bin] = acf_curve
    # If rolling analysis
    elif analysis_type == "rolling":
        periods = np.zeros(shape=(num_submovies, num_channels, total_bins))
        acfs = np.zeros(shape=(num_submovies, num_channels, total_bins, roll_size * 2 - 1))
        # Loop through submovies, channels, and bins
        its = num_submovies*num_channels*xpix*ypix
        with tqdm(total = its, miniters=its/100) as pbar:
            pbar.set_description( 'Periods: ')
            for submovie in range(num_submovies):
                for channel in range(num_channels):
                    for bin in range(total_bins):
                        pbar.update(1)
                        # Extract signal for rolling autocorrelation calculation
                        signal = bin_values[roll_by * submovie: roll_size + roll_by * submovie, channel, bin]
                        delay, acf_curve = create_acf_curves_calc_period(signal, num_frames_or_rollsize=roll_size, peak_thresh=peak_thresh)
                        periods[submovie, channel, bin] = delay
                        acfs[submovie, channel, bin] = acf_curve
                        
    return acfs, periods