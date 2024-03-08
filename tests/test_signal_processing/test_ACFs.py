'''import pytest
import pandas as pd
import numpy as np
from waveanalysis.signal_processing import calc_indv_standard_kymo_ACFs_periods

@pytest.fixture
def default_ACF_params():
    return {
        'num_channels': 2,
        'num_bins': 25,
        'num_frames': 391,
        'bin_values': np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/1_Group2_box_values.npy'),
        'analysis_type': 'standard',
        'peak_thresh': 0.1
    }

def test_ACFs_periods(default_ACF_params):
    known_periods = np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/1_Group2_periods.npy')
    known_acfs = np.load('/Users/domchom/Documents/GitHub/ZS_wave_analysis/tests/assets/standard/1_Group2_acfs.npy')
    assert isinstance(known_periods, np.ndarray)
    assert isinstance(known_acfs, np.ndarray)
    acfs, periods = calc_indv_standard_kymo_ACFs_periods(
        num_channels=default_ACF_params['num_channels'],
        num_bins=default_ACF_params['num_bins'],
        num_frames=default_ACF_params['num_frames'],
        bin_values=default_ACF_params['bin_values'],
        analysis_type=default_ACF_params['analysis_type'],
        peak_thresh=default_ACF_params['peak_thresh']
    )

    # have this here to just test 1_Group2.tif
    if acfs.shape == (2, 25):
        assert np.allclose(periods, known_periods), "Calculated periods do not match known periods."
        assert np.allclose(acfs, known_acfs), "Calculated ACFs do not match known ACFs."'''