import pytest
import numpy as np
from waveanalysis.image_props.image_properties import get_single_frame_properties
import json

@pytest.fixture
def default_filepaths():
    return [
        'tests/assets/kymo/1_Group1.tif',
        'tests/assets/kymo/1_Group2.tif'
    ]

def test_kymo_image_properties(default_filepaths):
    default_dicts = [
        'tests/assets/kymo/dicts_lists/1_Group1.tif_image_properties.json',
        'tests/assets/kymo/dicts_lists/1_Group2.tif_image_properties.json'
    ]
    # load csv
    for file_path in default_filepaths:
        with open(default_dicts[default_filepaths.index(file_path)], 'r') as file:
            known_results = json.load(file)
        exp_results = get_single_frame_properties(file_path)

        assert np.array_equal(known_results, exp_results)
