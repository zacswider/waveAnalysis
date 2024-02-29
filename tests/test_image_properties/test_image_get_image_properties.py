'''import pytest
import tifffile
import numpy as np
from waveanalysis.image_properties_signal.image_properties import get_standard_image_properties, get_kymo_image_properties, get_rolling_image_properties

@pytest.fixture
def test_standard_image_properties(
    image_path: str = '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/1_Group1.tif'
):
    # load tiff file
    with tifffile.TiffFile(image_path) as tif_file:
        metadata = tif_file.imagej_metadata
    known_num_channels = metadata.get('channels', 1)
    known_num_frames = metadata.get('frames', 1)
    
    return image_path, known_num_channels, known_num_frames

def test_get_standard_image_properties(test_standard_image_properties):
    image_path, known_num_channels, known_num_frames = test_standard_image_properties
    
    # Call the function to test
    exp_num_channels, exp_num_frames = get_standard_image_properties(image_path)

    assert known_num_channels == exp_num_channels
    assert known_num_frames == exp_num_frames'''