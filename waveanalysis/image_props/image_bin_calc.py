import numpy as np
import scipy.ndimage as nd
from scipy import signal as sig

def create_kymo_bin_array(
    image: np.ndarray,
    img_props: dict
) -> np.ndarray:
    
    line_width = img_props["line_width"]
    step = img_props["step"]
    num_channels = img_props["num_channels"]
    num_frames = img_props["num_frames"]
    num_columns = img_props["num_columns"]

    if line_width < 1:
        raise ValueError("Line width must be at least 1")
    
    # Calculate the total amount of bins based on the step size
    num_bins = (num_columns // step) 

    # Initialize array to store line values
    line_values = np.full(shape=(num_channels, num_bins, num_frames), fill_value=np.nan)

    for channel in range(num_channels):
        for col_num in range(0, num_columns, step):
            end_col = col_num + line_width
            if end_col <= num_columns:
                signal_slice = image[channel, :, col_num:end_col]
                if signal_slice.shape == (num_frames, line_width):
                    signal = np.mean(signal_slice, axis=1)
                    signal = sig.savgol_filter(signal, window_length=1, polyorder=0)
                    idx = col_num // step
                    line_values[channel, idx] = signal

    return line_values, num_bins

def create_multi_frame_bin_array(
    image: np.ndarray,
    img_props: dict
) -> np.ndarray:
    
    box_size = img_props["box_size"]
    step = img_props["step"]
    num_channels = img_props["num_channels"]
    num_frames = img_props["num_frames"]
    
    # Calculate the index for the center of the kernel
    ind = box_size // 2
    
    # Apply uniform filter to calculate mean signal over specified box size
    box_values = nd.uniform_filter(image[:, 0, :, :, :], size=(1, 1, box_size, box_size))[:, :, ind::step, ind::step]

    # Get the dimensions of the resulting mean image
    num_x_bins, num_y_bins = box_values.shape[-2:]
    num_bins = num_x_bins * num_y_bins
    box_values = box_values.reshape(num_frames, num_channels, num_bins)

    return box_values, num_bins, num_x_bins, num_y_bins