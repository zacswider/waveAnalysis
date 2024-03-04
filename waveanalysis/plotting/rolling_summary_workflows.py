import pandas as pd

from waveanalysis.plotting.rolling_plot_creation import return_mean_periods_shifts_props_plots

def plot_rolling_summary(
    num_channels: int,
    fullmovie_summary: pd.DataFrame,
    channel_combos: list[tuple[int, int]]
):
    rolling_mean_plots_dict = {}

    rolling_mean_periods = {}

    for channel in range(num_channels):
        rolling_mean_periods[f'Ch {channel + 1} Period'] = return_mean_periods_shifts_props_plots(
            independent_variable='Submovie',
            dependent_variable=f'Ch {channel + 1} Mean Period',
            dependent_error=f'Ch {channel + 1} StdDev Period',
            y_label=f'Ch {channel + 1} Mean ± StdDev Period (frames)',
            fullmovie_summary=fullmovie_summary
            )
            
    rolling_mean_plots_dict.update(rolling_mean_periods)

    rolling_mean_shifts = {}
    
    if num_channels > 1:
        for combo_number, combo in enumerate(channel_combos):
            rolling_mean_shifts[f'Ch{combo[0]+1}-Ch{combo[1]+1} Shift'] = return_mean_periods_shifts_props_plots(
                independent_variable='Submovie',
                dependent_variable=f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean Shift',
                dependent_error=f'Ch{combo[0]+1}-Ch{combo[1]+1} StdDev Shift',
                y_label=f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean ± StdDev Shift (frames)',
                fullmovie_summary=fullmovie_summary
                )
            
    rolling_mean_plots_dict.update(rolling_mean_shifts)

    rolling_mean_peak_props = {}

    for channel in range(num_channels):
        for prop_name in ['Width', 'Max', 'Min', 'Amp']:
            rolling_mean_peak_props[f'Ch{channel+1} {prop_name}'] = return_mean_periods_shifts_props_plots(
                independent_variable='Submovie',
                dependent_variable=f'Ch {channel+1} Mean Peak {prop_name}',
                dependent_error=f'Ch {channel+1} StdDev Peak {prop_name}',
                y_label=f'Ch {channel+1} Mean ± StdDev Peak {prop_name} (frames)',
                fullmovie_summary=fullmovie_summary
                )
                    
    rolling_mean_plots_dict.update(rolling_mean_peak_props)

    return rolling_mean_plots_dict