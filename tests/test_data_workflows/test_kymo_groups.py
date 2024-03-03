import pytest
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows.kymograph_workflow import kymograph_workflow

@pytest.fixture
def default_log_params():
    return {
        'Line Size(px)': 5,
        'Line Shift(px)': 5,
        'Base Directory': 'tests/assets/kymo',
        'ACF Peak Prominence': 0.1,
        'Group Names': ['Group1, Group2'],
        'Plot Summary ACFs': False,
        'Plot Summary CCFs': True,
        'Plot Summary Peaks': False,
        'Plot Individual ACFs': False,
        'Plot Individual CCFs': False,
        'Plot Individual Peaks': False,
        'Group Matching Errors': [],
        'Files Processed': [],
        'Files Not Processed': [],
        'Plotting errors': [],
    }

def test_kymo(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/kymo/kymo_known_results.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = kymograph_workflow(
        folder_path=str(Path('tests/assets/kymo/')),
        group_names=['Group1','Group2'],
        log_params=default_log_params,
        analysis_type='kymograph',
        box_shift=default_log_params['Line Shift(px)'],
        line_width=default_log_params['Line Size(px)'],         
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        plot_summary_ACFs=default_log_params['Plot Summary ACFs'],
        plot_summary_CCFs=default_log_params['Plot Summary CCFs'],
        plot_summary_peaks=default_log_params['Plot Summary Peaks'],
        plot_indv_ACFs=default_log_params['Plot Individual ACFs'],
        plot_indv_CCFs=default_log_params['Plot Individual CCFs'],
        plot_indv_peaks=default_log_params['Plot Individual Peaks'],
    )
    assert pd.testing.assert_frame_equal(known_results, exp_results) is None
