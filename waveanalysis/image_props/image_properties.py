import numpy as np
import tifffile

def get_multi_frame_properties(image_path: str):
    with tifffile.TiffFile(image_path) as tif_file:
        tags = tif_file.pages[0].tags
        metadata = tif_file.imagej_metadata

        x = get_voxel_size(tags, 'XResolution')
        y = get_voxel_size(tags, 'YResolution')
        z = metadata.get('spacing', 1.0)
        pixel_size = [x, y, z]

        num_channels = metadata.get('channels')
        frame_interval = metadata.get('finterval', np.nan)
        pixel_unit = metadata.get('unit')
        num_frames = metadata.get('frames')
        
    return num_channels, num_frames, frame_interval, pixel_size, pixel_unit

def get_single_frame_properties(image_path: str, image: np.ndarray):
    with tifffile.TiffFile(image_path) as tif_file:
        tags = tif_file.pages[0].tags
        metadata = tif_file.imagej_metadata

        x = get_voxel_size(tags, 'XResolution')
        y = get_voxel_size(tags, 'YResolution')
        z = metadata.get('spacing', 1.0)
        pixel_size = [x, y, z]
        
        num_channels = metadata.get('channels')
        frame_interval = metadata.get('finterval')
        pixel_unit = metadata.get('unit')
        num_frames = image.shape[-2]
        num_columns = image.shape[-1]
        
    return num_channels, num_columns, num_frames, frame_interval, pixel_size, pixel_unit

def get_voxel_size(tags, key):
    assert key in ['XResolution', 'YResolution']
    if key in tags:
        num_pixels, units = tags[key].value
        return units / num_pixels
    else:
        return 1.0