import numpy as np
import scipy.ndimage as nd
from scipy import signal as sig

def create_kymo_signals(
    line_width: int,
    total_columns: int,
    step: int,
    num_channels: int,
    num_frames: int,
    image: np.ndarray
) -> np.ndarray:

    if line_width < 1:
        raise ValueError("Line width must be at least 1")
    
    # Calculate the total amount of bins based on the step size
    total_bins = (total_columns // step) 

    # Initialize array to store line values
    line_values = np.full(shape=(num_channels, total_bins, num_frames), fill_value=np.nan)

    for channel in range(num_channels):
        for col_num in range(0, total_columns, step):
            end_col = col_num + line_width
            if end_col <= total_columns:
                signal_slice = image[channel, :, col_num:end_col]
                if signal_slice.shape == (num_frames, line_width):
                    signal = np.mean(signal_slice, axis=1)
                    signal = sig.savgol_filter(signal, window_length=1, polyorder=0)
                    idx = col_num // step
                    line_values[channel, idx] = signal

    return line_values, total_bins

def create_standard_signals(
    kernel_size: int, 
    step: int, 
    num_channels: int, 
    num_frames: int, 
    image: np.ndarray
) -> np.ndarray:
    
    # Calculate the index for the center of the kernel
    ind = kernel_size // 2
    # Apply uniform filter to calculate mean signal over specified box size
    box_values = nd.uniform_filter(image[:, 0, :, :, :], size=(1, 1, kernel_size, kernel_size))[:, :, ind::step, ind::step]
    # Get the dimensions of the resulting mean image
    xpix, ypix = box_values.shape[-2:]
    # We are either binning the image into boxes (standard) or columns (kymographs), so just call bins for simplicity
    total_bins = xpix * ypix
    box_values = box_values.reshape(num_frames, num_channels, total_bins)

    return box_values, total_bins, xpix, ypix