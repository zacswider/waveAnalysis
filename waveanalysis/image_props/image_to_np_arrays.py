import tifffile
import pathlib
import numpy as np

def tiff_to_np_array_single_frame(folder_path: str) -> dict:
    """
    Convert TIFF files in a folder to a dictionary of numpy arrays.

    Args:
        folder_path (str): The path to the folder containing the TIFF files.

    Returns:
        dict: A dictionary where the keys are the file names and the values are the corresponding numpy arrays.
    """
    input_path = pathlib.Path(folder_path)
    images = {}

    for file_path in input_path.glob('*.tif'):
        # Load the TIFF file into a numpy array
        image = tifffile.imread(file_path)

        with tifffile.TiffFile(file_path) as tif_file:
            metadata = tif_file.imagej_metadata
        num_channels = metadata.get('channels', 1)

        image = image.reshape(num_channels, 
                                image.shape[-2],  # cols
                                image.shape[-1])  # rows
            
        images[file_path.name] = image

    # Sort the dictionary keys alphabetically
    images = {key: images[key] for key in sorted(images)}

    return images

def tiff_to_np_array_multi_frame(folder_path: str) -> dict:
    '''
    Convert TIFF files in a folder to a dictionary of numpy arrays.
    
    Args:
        folder_path (str): The path to the folder containing the TIFF files.
        
    Returns:
        dict: A dictionary where the keys are the file names and the values are the corresponding numpy arrays.
    '''
    input_path = pathlib.Path(folder_path)
    images = {}

    for file_path in input_path.glob('*.tif'):
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

        images[file_path.name] = image

    # Sort the dictionary keys alphabetically
    images = {key: images[key] for key in sorted(images)}

    return images