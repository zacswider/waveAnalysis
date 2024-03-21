import pytest
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows.combined_workflow import combined_workflow

@pytest.fixture
def default_log_params():
    return {
        'Box Size(px)': 20,
        'Box Shift(px)': 20,
        'Base Directory': 'tests/assets/standard',
        'ACF Peak Prominence': 0.1,
        'Group Names': ['Group1, Group2'],
        'Plot Summary ACFs': True,
        'Plot Summary CCFs': True,
        'Plot Summary Peaks': True,
        'Plot Individual ACFs': False,
        'Plot Individual CCFs': False,
        'Plot Individual Peaks': False,
        'Calc Wave Speeds': False,
        'Plot Wave Speeds': False,
        'Files Processed': [],
        'Files Not Processed': [],
        'Errors': [],
        'Frame Interval': [],
        'Pixel Size': []
    }

def test_standard(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/standard/standard_known_results.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = combined_workflow(
        folder_path=str(Path('tests/assets/standard/')),
        group_names=['Group1','Group2'],
        log_params=default_log_params,
        analysis_type='standard',
        box_size=default_log_params['Box Size(px)'],
        box_shift=default_log_params['Box Shift(px)'],
        line_width=None, #type: ignore
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        plot_summary_ACFs=default_log_params['Plot Summary ACFs'],
        plot_summary_CCFs=default_log_params['Plot Summary CCFs'],
        plot_summary_peaks=default_log_params['Plot Summary Peaks'],
        plot_indv_ACFs=default_log_params['Plot Individual ACFs'],
        plot_indv_CCFs=default_log_params['Plot Individual CCFs'],
        plot_indv_peaks=default_log_params['Plot Individual Peaks'],
        calc_wave_speeds=None, #type: ignore
        plot_wave_speeds=None, #type: ignore
    )
    assert pd.testing.assert_frame_equal(known_results, exp_results) is None