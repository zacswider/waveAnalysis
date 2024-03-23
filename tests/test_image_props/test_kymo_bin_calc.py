import pytest
import numpy as np
from waveanalysis.image_props.image_bin_calc import create_kymo_bin_array

@pytest.fixture
def default_bin_values():
    return [
        np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/kymo/numpy_arrays/kymo_1_Group1.tif_bin_values.npy'),
        np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/kymo/numpy_arrays/kymo_1_Group2.tif_bin_values.npy')
    ]
     
def test_kymo_bin_calc(default_bin_values):
    default_array_paths = [
        np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/kymo/numpy_arrays/kymo_1_Group1_array.npy'),
        np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/kymo/numpy_arrays/kymo_1_Group2_array.npy')
    ]

    default_array_pathss = [
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/kymo/numpy_arrays/kymo_1_Group1_array.npy',
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/kymo/numpy_arrays/kymo_1_Group2_array.npy'
    ]

    default_dicts = {
        "Group1": 
            {"num_channels": 2, 
            "num_columns": 37, 
            "num_frames": 87, 
            "frame_interval": 5.4, 
            "pixel_size": [0.2661448807564476, 9.400086480795624, 1.0], 
            "pixel_unit": "micron",
            "line_width": 5,
            "step": 5
            }
        ,

        "Group2":
            {"num_channels": 2, 
            "num_columns": 58, 
            "num_frames": 157,
            "frame_interval": 3.84, 
            "pixel_size": [0.2661448807564476, 1.0, 1.0],
            "pixel_unit": "micron",
            "line_width": 5,
            "step": 5
            }
        }
    
    assert False