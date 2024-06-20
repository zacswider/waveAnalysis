'''import pytest
import numpy as np
from waveanalysis.signal_processing.peak_properties import calc_indv_peak_props_workflow

import pickle
import json

@pytest.fixture
def default_peak_props():
    return [
        'tests/assets/standard/dicts_lists/peak_props_1_Group1.tif.pkl',
        'tests/assets/standard/dicts_lists/peak_props_1_Group2.tif.pkl'
        ]
     
def test_peak_props_calc(default_peak_props):

    default_bin_values = [
        np.load('tests/assets/standard/numpy_arrays/standard_1_Group1.tif_bin_values.npy'),
        np.load('tests/assets/standard/numpy_arrays/standard_1_Group2.tif_bin_values.npy')
        ]

    default_dicts = [
        'tests/assets/standard/dicts_lists/standard_image_properties_1_Group1_final.json',
        'tests/assets/standard/dicts_lists/standard_image_properties_1_Group2_final.json'
    ]

    for bin_values, peak_prop_file, img_props_file in zip(default_bin_values, default_peak_props, default_dicts):
        # Load the pickle file
        with open(peak_prop_file, 'rb') as f:
            known_results = pickle.load(f)
        with open(img_props_file, 'r') as file:
            img_props_dict = json.load(file)
        _, _, _, _, exp_results = calc_indv_peak_props_workflow(bin_values, img_props_dict)

        for key, value in known_results.items():
            for new_key, new_value in value.items():
                assert np.array_equal(new_value, exp_results[key][new_key], equal_nan=True)
            
'''