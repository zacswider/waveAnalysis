import matplotlib.pyplot as plt
import pandas as pd

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
        