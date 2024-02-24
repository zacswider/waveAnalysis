import os
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def make_log(
    directory: str, 
    logParams: dict
):
    '''
    Convert dictionary of parameters to a log file and save it in the directory
    '''
    now = datetime.datetime.now()
    logPath = os.path.join(directory, f"!log-{now.strftime('%Y%m%d%H%M')}.txt")
    logFile = open(logPath, "w")                                    
    logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")     
    for key, value in logParams.items():                            
        logFile.write('%s: %s\n' % (key, value))                    
    logFile.close()           


def generate_group_comparison(
    main_save_path: str,
    processor: object,
    summary_df: pd.DataFrame,
    log_params: dict
):
    print('Generating group comparisons...')
    # make a group comparisons save path in the main save directory
    group_save_path = os.path.join(main_save_path, "!groupComparisons")
    if not os.path.exists(group_save_path):
        os.makedirs(group_save_path)
    
    # make a list of parameters to compare
    stats_to_compare = ['Mean']
    channels_to_compare = [f'Ch {i+1}' for i in range(processor.num_channels)]
    measurements_to_compare = ['Period', 'Shift', 'Peak Width', 'Peak Max', 'Peak Min', 'Peak Amp', 'Peak Rel Amp']
    params_to_compare = []
    for channel in channels_to_compare:
        for stat in stats_to_compare:
            for measurement in measurements_to_compare:
                params_to_compare.append(f'{channel} {stat} {measurement}')

    # will compare the shifts if multichannel movie
    if hasattr(processor, 'channel_combos'):
        shifts_to_compare = [f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean Shift' for combo in processor.channel_combos]
        params_to_compare.extend(shifts_to_compare)

    # generate and save figures for each parameter
    for param in params_to_compare:
        try:
            fig = plotComparisons(summary_df, param)
            fig.savefig(f'{group_save_path}/{param}.png')  # type: ignore
            plt.close(fig)
        except ValueError:
            log_params['Plotting errors'].append(f'No data to compare for {param}')


def plotComparisons(
    dataFrame: pd.DataFrame, 
    dependent: str, 
    independent = 'Group Name'
) -> plt.Figure:
        '''
        This func accepts a dataframe, the name of a dependent variable, and the name of an
        independent variable (by default, set to Group Name). It returns a figure object showing
        a box and scatter plot of the dependent variable grouped by the independent variable.
        '''
        ax = sns.boxplot(x=independent, y=dependent, data=dataFrame, palette = "Set2", showfliers = False)
        ax = sns.swarmplot(x=independent, y=dependent, data=dataFrame, color=".25")	
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        fig = ax.get_figure()
        return fig