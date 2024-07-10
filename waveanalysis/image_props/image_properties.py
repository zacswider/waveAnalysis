import tifffile

def get_multi_frame_properties(image_path: str) -> dict:
    """
    Retrieves the properties of a multi-frame image.

    Args:
        image_path (str): The path to the multi-frame image file.

    Returns:
        dict: A dictionary containing the properties of the multi-frame image.
            - num_channels (int): The number of channels in the image.
            - num_frames (int): The number of frames in the image.
            - frame_interval (float): The time interval between frames.
            - pixel_size (list): The size of each pixel in the image, in the X, Y, and Z dimensions.
            - pixel_unit (str): The unit of measurement for the pixel size.
    """
    with tifffile.TiffFile(image_path) as tif_file:
        tags = tif_file.pages[0].tags
        metadata = tif_file.imagej_metadata

        # Get the pixel size in the X, Y, and Z dimensions
        x = get_voxel_size(tags, 'XResolution')
        y = get_voxel_size(tags, 'YResolution')
        z = metadata.get('spacing', 1.0)
        pixel_size = [x, y, z]

        # Get the number of channels, frames, pixel unit, and the frame interval
        num_channels = metadata.get('channels', 1)
        frame_interval = metadata.get('finterval', 1)
        pixel_unit = metadata.get('unit', 'px')
        num_frames = metadata.get('frames', 1)
    
    img_props_dict = {
        'num_channels': num_channels,
        'num_frames': num_frames,
        'frame_interval': frame_interval,
        'pixel_size': pixel_size,
        'pixel_unit': pixel_unit
    }
        
    return img_props_dict

def get_single_frame_properties(image_path: str) -> dict:

    image = tifffile.imread(image_path)

    with tifffile.TiffFile(image_path) as tif_file:
        
        tags = tif_file.pages[0].tags
        metadata = tif_file.imagej_metadata

        # Get the pixel size in the X, Y, and Z dimensions
        x = get_voxel_size(tags, 'XResolution')
        y = get_voxel_size(tags, 'YResolution')
        z = metadata.get('spacing', 1.0)
        pixel_size = [x, y, z]
        
        # Get the number of channels, frame interval, and pixel unit
        num_channels = metadata.get('channels', 1)
        frame_interval = metadata.get('finterval', 1)
        pixel_unit = metadata.get('unit', 'px')

    # Get the dimensions of the image
    num_frames = image.shape[-2]
    num_columns = image.shape[-1]

    img_props_dict = {
        'num_channels': num_channels,
        'num_columns': num_columns,
        'num_frames': num_frames,
        'frame_interval': frame_interval,
        'pixel_size': pixel_size,
        'pixel_unit': pixel_unit
    }
        
    return img_props_dict

def get_voxel_size(tags, key) -> float:
    '''
    Get the size of each pixel in the image.
    '''
    # Check if the key is in the tags
    assert key in ['XResolution', 'YResolution']
    # Get the number of pixels and the units
    if key in tags:
        num_pixels, units = tags[key].value
        return units / num_pixels
    else: # if the key is not in the tags, return 1.0
        return 1.0