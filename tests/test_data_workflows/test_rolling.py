'''import pytest
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows import rolling_workflow

@pytest.fixture
def default_log_params():
    return {
        'Box Size(px)': 20,
        'Box Shift(px)': 20,
        'Base Directory': 'tests/assets/rolling',
        'ACF Peak Prominence': 0.1,
        'Files Processed': [],
        'Files Not Processed': [],
        'Plotting errors': [],
        'Submovies Used' : [],
        'Errors': [],
        'Frame Interval': [],
        'Pixel Size': []
        }


def test_rolling(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/rolling/1_Group2_summary.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = rolling_workflow(
        folder_path=str(Path('tests/assets/rolling/')),
        log_params=default_log_params,
        box_size=default_log_params['Box Size(px)'],
        box_shift=default_log_params['Box Shift(px)'],
        roll_size=50,
        roll_by=5,       
        acf_peak_thresh=default_log_params['ACF Peak Prominence']
    )
    assert pd.testing.assert_frame_equal(known_results, exp_results) is None'''