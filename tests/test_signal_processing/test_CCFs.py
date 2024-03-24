import pytest
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_CCF_workflow

import pickle
import json

@pytest.fixture
def default_CCFs():
    return [
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/CCF_1_Group1.tif.pkl',
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/CCF_1_Group2.tif.pkl'
    ]
     
def test_CCF_calc(default_CCFs):
    default_bin_values = [
        np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/numpy_arrays/standard_1_Group1.tif_bin_values.npy'),
        np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/numpy_arrays/standard_1_Group2.tif_bin_values.npy')
        ]

    default_dicts = [
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/standard_image_properties_1_Group1_final.json',
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/standard_image_properties_1_Group2_final.json'
    ]

    for bin_values, ccf_file, img_props_file in zip(default_bin_values, default_CCFs, default_dicts):
        # Load the pickle file
        with open(ccf_file, 'rb') as f:
            known_results = pickle.load(f)
        with open(img_props_file, 'r') as file:
            img_props_dict = json.load(file)
        exp_results = calc_indv_CCF_workflow(bin_values, img_props_dict)

        assert np.array_equal(known_results, exp_results)