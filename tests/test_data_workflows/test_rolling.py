import pytest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows.rolling_workflow import rolling_workflow

# TODO: come up with a better way to test rolling_workflow. Need to use a summary file from one of the movies to more accurately test the function.

@pytest.fixture(autouse=True)
def ignore_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield

@pytest.fixture
def default_log_params():
    return {
        'Box Size(px)': 20,
        'Box Shift(px)': 20,
        'Base Directory': 'tests/assets/rolling',
        'ACF Peak Prominence': 0.1,
        'Group Matching Errors': [],
        'Files Processed': [],
        'Files Not Processed': [],
        'Plotting errors': [],
        'Submovies Used' : []
    }

def test_rolling(default_log_params):
    # load csv
    # known_results = pd.read_csv('tests/assets/rolling/rolling_known_results.csv')
    # assert isinstance(known_results, pd.DataFrame)
    exp_results = rolling_workflow(
        folder_path=str(Path('tests/assets/rolling/')),
        log_params=default_log_params,
        analysis_type='rolling',
        box_size=default_log_params['Box Size(px)'],
        box_shift=default_log_params['Box Shift(px)'],
        subframe_size=20,
        subframe_roll=5,       
        line_width=np.nan,          # type: ignore ;not part of standard analysis
        acf_peak_thresh=default_log_params['ACF Peak Prominence']
    )
    # assert pd.testing.assert_frame_equal(known_results, exp_results) is None

