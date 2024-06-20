import pytest
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_shift_workflow

import pickle
import json

@pytest.fixture
def default_shifts():
    return [
        'tests/assets/standard/dicts_lists/shifts_1_Group1.tif.pkl',
        'tests/assets/standard/dicts_lists/shifts_1_Group2.tif.pkl'
    ]
     
def test_shift_calc(default_shifts):
    default_period_values = [
        'tests/assets/standard/dicts_lists/periods_1_Group1.tif.pkl',
        'tests/assets/standard/dicts_lists/periods_1_Group2.tif.pkl'
        ]
    
    default_ccf_values = [
        'tests/assets/standard/dicts_lists/CCF_1_Group1.tif.pkl',
        'tests/assets/standard/dicts_lists/CCF_1_Group2.tif.pkl'
        ]

    default_dicts = [
        'tests/assets/standard/dicts_lists/standard_image_properties_1_Group1_final.json',
        'tests/assets/standard/dicts_lists/standard_image_properties_1_Group2_final.json'
    ]

    for period_file, ccf_file, shift_file, img_props_file in zip(default_period_values, default_ccf_values, default_shifts, default_dicts):
        # Load the pickle file
        with open(period_file, 'rb') as f:
            periods = pickle.load(f)
        with open(ccf_file, 'rb') as f:
            ccfs = pickle.load(f)
        with open(img_props_file, 'r') as file:
            img_props_dict = json.load(file)
        with open(shift_file, 'rb') as f:
            known_results = pickle.load(f)

        exp_results = calc_indv_shift_workflow(ccfs, periods, img_props_dict)

        assert np.array_equal(known_results, exp_results)