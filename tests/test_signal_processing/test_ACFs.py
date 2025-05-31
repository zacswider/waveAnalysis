import pytest
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_ACF_workflow

import pickle
import json

@pytest.fixture
def default_ACFs():
    return [
        'tests/assets/standard/dicts_lists/1_Group1.tif_indv_acfs.pkl',
        'tests/assets/standard/dicts_lists/1_Group2.tif_indv_acfs.pkl'
    ]
     
def test_ACF_calc(default_ACFs):
    default_bin_values = [
        np.load('tests/assets/standard/numpy_arrays/standard_1_Group1.tif_bin_values.npy'),
        np.load('tests/assets/standard/numpy_arrays/standard_1_Group2.tif_bin_values.npy')
        ]

    default_dicts = [
        'tests/assets/standard/dicts_lists/1_Group1_img_props.json',
        'tests/assets/standard/dicts_lists/1_Group2_img_props.json'
    ]

    for bin_values, acf_file, img_props_file in zip(default_bin_values, default_ACFs, default_dicts):
        # Load the pickle file
        with open(acf_file, 'rb') as f:
            known_results = pickle.load(f)
        with open(img_props_file, 'r') as file:
            img_props_dict = json.load(file)
        exp_results = calc_indv_ACF_workflow(bin_values, img_props_dict)

        np.testing.assert_allclose(
            known_results,
            exp_results,
            equal_nan=True,
            atol=1e-1,
        )