import pytest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows.combined_workflow import combined_workflow

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
        'Submovies Used' : [],
        'Plot Summary ACFs': False,
        'Plot Summary CCFs': False,
        'Plot Summary Peaks': False,
        'Plot Individual ACFs': False,
        'Plot Individual CCFs': False,
        'Plot Individual Peaks': False
        }


def test_combined(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/rolling/1_Group2_summary.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = combined_workflow(
        folder_path=str(Path('tests/assets/rolling/')),
        group_names=[''],
        log_params=default_log_params,
        analysis_type='rolling',
        box_size=default_log_params['Box Size(px)'],
        box_shift=default_log_params['Box Shift(px)'],
        subframe_size=50,
        subframe_roll=5,       
        line_width=np.nan,          # type: ignore ;not part of standard analysis
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        plot_summary_ACFs=default_log_params['Plot Summary ACFs'],
        plot_summary_CCFs=default_log_params['Plot Summary CCFs'],
        plot_summary_peaks=default_log_params['Plot Summary Peaks'],
        plot_ind_ACFs=np.nan,          # type: ignore ;not part of standard analysis
        plot_ind_CCFs=np.nan,          # type: ignore ;not part of standard analysis
        plot_ind_peaks=np.nan,          # type: ignore ;not part of standard analysis
    )
    assert pd.testing.assert_frame_equal(known_results, exp_results) is None