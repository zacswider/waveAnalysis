import matplotlib.pyplot as plt
import pandas as pd

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
            y_label=f'Ch {channel + 1} Mean ± StdDev Period (seconds)',
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
                y_label=f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean ± StdDev Shift (seconds)',
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
                y_label=f'Ch {channel+1} Mean ± StdDev Peak {prop_name} (seconds)',
                fullmovie_summary=fullmovie_summary
                )
                    
    rolling_mean_plots_dict.update(rolling_mean_peak_props)

    return rolling_mean_plots_dict

def return_mean_periods_shifts_props_plots(
    independent_variable: str, 
    dependent_variable: str, 
    dependent_error: str, 
    y_label: str,
    fullmovie_summary: pd.DataFrame
) -> plt.Figure:    
    '''
    Space saving function to generate the rolling summary plots'''      
    fig, ax = plt.subplots()

    # plot the dataframe
    ax.plot(fullmovie_summary[independent_variable], 
            fullmovie_summary[dependent_variable])
    
    # fill between the ± standard deviation of the dependent variable
    ax.fill_between(x = fullmovie_summary[independent_variable],
                    y1 = fullmovie_summary[dependent_variable] - fullmovie_summary[dependent_error],
                    y2 = fullmovie_summary[dependent_variable] + fullmovie_summary[dependent_error],
                    color = 'blue',
                    alpha = 0.25)

    # set axis labels
    ax.set_xlabel('Frame Number')
    ax.set_ylabel(y_label)
    ax.set_title(f'{y_label} over time')
    
    plt.close(fig)
    return fig