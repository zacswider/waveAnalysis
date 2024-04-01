import tifffile
import numpy as np

def tiff_to_np_array_single_frame(file_path: str) -> np.ndarray:
    """
    Convert a TIFF file to a NumPy array representing a single frame.

    Args:
        file_path (str): The path to the TIFF file.

    Returns:
        np.ndarray: A NumPy array representing the image data of a single frame.
    """
    image = tifffile.imread(file_path)

    with tifffile.TiffFile(file_path) as tif_file:
        metadata = tif_file.imagej_metadata
    num_channels = metadata.get('channels', 1)

    image = image.reshape(num_channels, 
                            image.shape[-2],  # cols
                            image.shape[-1])  # rows
    
    return image

def tiff_to_np_array_multi_frame(file_path: str) -> np.ndarray:
    """
    Convert a multi-frame TIFF file to a numpy array.

    Args:
        file_path (str): The path to the TIFF file.

    Returns:
        np.ndarray: The numpy array representing the TIFF file.
    """
    # Load the TIFF file into a numpy array
    image = tifffile.imread(file_path)

    with tifffile.TiffFile(file_path) as tif_file:
        metadata = tif_file.imagej_metadata
    num_channels = metadata.get('channels', 1)
    num_frames = metadata.get('frames', 1)
    num_slices = metadata.get('slices', 1)

    # Max project if multiple slices
    if num_slices > 1:
        print('Max projecting image stack')
        image = np.max(image, axis=1)
        num_slices = 1
        
    image = image.reshape(num_frames, 
                        num_slices, 
                        num_channels, 
                        *image.shape[-2:])

    return image