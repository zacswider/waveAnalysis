import numpy as np
import tifffile

def get_image_properties(
    image_path: str
):
    with tifffile.TiffFile(image_path) as tif_file:
        metadata = tif_file.imagej_metadata
    num_channels = metadata.get('channels', 1)
    num_frames = metadata.get('frames', 1)

    return num_channels, num_frames

def get_kymo_image_properties(
    image_path: str,
    image: np.ndarray
):
    with tifffile.TiffFile(image_path) as tif_file:
        metadata = tif_file.imagej_metadata
    num_channels = metadata.get('channels', 1)
    total_columns = image.shape[-1]
    num_frames = image.shape[-2]

    return num_channels, total_columns, num_frames