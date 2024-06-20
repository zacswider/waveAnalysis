import pytest
from waveanalysis.plotting.mean_plot_creation import plot_mean_ACF_workflow

import pickle
import json

@pytest.fixture
def default_mean_ACF_plots():
    return [
        'tests/assets/standard/dicts_lists/mean_acf_figs_1_Group1.tif.pkl',
        'tests/assets/standard/dicts_lists/mean_acf_figs_1_Group2.tif.pkl'
        ]
     
def test_mean_ACF_plot(default_mean_ACF_plots):
    default_img_params = [
        'tests/assets/standard/dicts_lists/img_parameters_1_Group1.tif.pkl',
        'tests/assets/standard/dicts_lists/img_parameters_1_Group2.tif.pkl'       
        ]

    default_dicts = [
        'tests/assets/standard/dicts_lists/standard_image_properties_1_Group1_final.json',
        'tests/assets/standard/dicts_lists/standard_image_properties_1_Group2_final.json'
    ]

    default_ACFs = [
        'tests/assets/standard/dicts_lists/ACF_1_Group1.tif.pkl',
        'tests/assets/standard/dicts_lists/ACF_1_Group2.tif.pkl'
    ]

    for img_param_file, acf_file, img_props_file, mean_acf_plot_file in zip(default_img_params, default_ACFs, default_dicts, default_mean_ACF_plots):
        # Load the pickle file
        with open(acf_file, 'rb') as f:
            acf_curves = pickle.load(f)
        with open(img_props_file, 'r') as file:
            img_props_dict = json.load(file)
        with open(img_param_file, 'rb') as f:
            img_params = pickle.load(f)
        with open(mean_acf_plot_file, 'rb') as f:
            known_results = pickle.load(f)
        exp_results = plot_mean_ACF_workflow(img_params, img_props_dict, acf_curves)

        
        assert len(exp_results) == len(known_results)
        # TODO: figure out a better way to test this
