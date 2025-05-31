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
        'Group Names': ['Group1', 'Group2'], #['DC191', 'DC192', 'DC193', 'DC206'], # #['WT','Y653A','F649A','FYAA','FY-AA_P731D','FY-AA_PC-DK'], # # #
        'Plot Summary ACFs': False,
        'Plot Summary CCFs': False,
        'Plot Summary Peaks': False,
        'Plot Individual ACFs': False,
        'Plot Individual CCFs': False,
        'Plot Individual Peaks': False,
        'Calc Wave Speeds': False,
        'Plot Wave Speeds': False,
        'Files Processed': [],
        'Files Not Processed': [],
        'Errors': [],
        'Frame Interval': [],
        'Pixel Size': [],
        'Small Shifts Correction': True,
        'CCF Peak Prominence': 0.1,
    }

def test_standard_workflow(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/standard/known_standard_summary.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = combined_workflow(
        folder_path=str(Path('tests/assets/standard/')),
        group_names= default_log_params['Group Names'],
        log_params=default_log_params,
        analysis_type='standard',
        box_size=default_log_params['Box Size(px)'],
        bin_shift=default_log_params['Box Shift(px)'],
        line_width=None, #type: ignore
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        ccf_peak_thresh=default_log_params['CCF Peak Prominence'],
        small_shifts_correction=default_log_params['Small Shifts Correction'],
        plot_summary_ACFs=default_log_params['Plot Summary ACFs'],
        plot_summary_CCFs=default_log_params['Plot Summary CCFs'],
        plot_summary_peaks=default_log_params['Plot Summary Peaks'],
        plot_indv_ACFs=default_log_params['Plot Individual ACFs'],
        plot_indv_CCFs=default_log_params['Plot Individual CCFs'],
        plot_indv_peaks=default_log_params['Plot Individual Peaks'],
        calc_wave_speeds=None, #type: ignore
        plot_wave_speeds=None, #type: ignore
        test=True
    )
    # assert pd.testing.assert_frame_equal(known_results, exp_results) is None
    pd.testing.assert_frame_equal(
        known_results.reset_index(drop=True),
        exp_results.reset_index(drop=True),
        atol=1e-1,
    )