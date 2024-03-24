import pytest
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_period_workflow

import pickle
import json

@pytest.fixture
def default_periods():
    return [
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/periods_1_Group1.tif.pkl',
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/periods_1_Group2.tif.pkl'
    ]
     
def test_period_calc(default_periods):
    default_acf_values = [
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/ACF_1_Group1.tif.pkl',
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/ACF_1_Group2.tif.pkl'
        ]

    default_dicts = [
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/standard_image_properties_1_Group1_final.json',
        '/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/dicts_lists/standard_image_properties_1_Group2_final.json'
    ]

    for acf_array, period_file, img_props_file in zip(default_acf_values, default_periods, default_dicts):
        # Load the pickle file
        with open(period_file, 'rb') as f:
            known_results = pickle.load(f)
        with open(acf_array, 'rb') as f:
            acf_array = pickle.load(f)
        with open(img_props_file, 'r') as file:
            img_props_dict = json.load(file)
        exp_results = calc_indv_period_workflow(acf_array, img_props_dict)

        assert np.array_equal(known_results, exp_results)