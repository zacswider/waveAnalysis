import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows.combined import combined_workflow

@pytest.fixture
def default_log_params():
    return {
        'Box Size(px)': 20,
        'Box Shift(px)': 20,
        'Base Directory': 'tests/assets/standard',
        'ACF Peak Prominence': 0.1,
        'Group Names': ['Group1, Group2'],
        'Plot Summary ACFs': False,
        'Plot Summary CCFs': False,
        'Plot Summary Peaks': False,
        'Plot Individual ACFs': False,
        'Plot Individual CCFs': False,
        'Plot Individual Peaks': False,
        'Group Matching Errors': [],
        'Files Processed': [],
        'Files Not Processed': [],
        'Plotting errors': [],
    }

# TODO: define default kymograph log params
# TODO: define default rolling log params
# TODO: create known output for kymograph analysis to test against
# TODO: create known output for rolling analysis to test against

def test_combined(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/standard/standard_known_results.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = combined_workflow(
        folder_path=str(Path('tests/assets/standard/')),
        group_names=[''],
        log_params=default_log_params,
        analysis_type='standard',
        box_size=default_log_params['Box Size(px)'],
        box_shift=default_log_params['Box Shift(px)'],
        subframe_size=np.nan,       # type: ignore ; not part of standard analysis
        subframe_roll=np.nan,       # type: ignore ;not part of standard analysis
        line_width=np.nan,          # type: ignore ;not part of standard analysis
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        plot_summary_ACFs=default_log_params['Plot Summary ACFs'],
        plot_summary_CCFs=default_log_params['Plot Summary CCFs'],
        plot_summary_peaks=default_log_params['Plot Summary Peaks'],
        plot_ind_ACFs=default_log_params['Plot Individual ACFs'],
        plot_ind_CCFs=default_log_params['Plot Individual CCFs'],
        plot_ind_peaks=default_log_params['Plot Individual Peaks'],
    )
    assert pd.testing.assert_frame_equal(known_results, exp_results) is None
