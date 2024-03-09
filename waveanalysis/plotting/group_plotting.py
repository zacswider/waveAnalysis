import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_group_comparison(
    summary_df: pd.DataFrame,
    log_params: dict
):
    print('Generating group comparisons...')
    
    mean_parameter_figs = {}
    parameters_to_compare = [column for column in summary_df.columns if 'Mean' in column]

    # generate and save figures for each parameter
    for param in parameters_to_compare:
        try:
            ax = sns.boxplot(x='Group Name', 
                            y=param, 
                            data=summary_df, 
                            palette = "Set2", 
                            showfliers = False)
            ax = sns.swarmplot(x='Group Name', 
                            y=param, 
                            data=summary_df, 
                            color=".25")	
            ax.set_xticklabels(ax.get_xticklabels(), 
                            rotation=45)
            fig = ax.get_figure()
            
            mean_parameter_figs[param] = fig
            plt.close(fig)

        except ValueError:
            log_params['Plotting errors'].append(f'No data to compare for {param}')

    return mean_parameter_figs