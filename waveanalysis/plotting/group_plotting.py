import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_group_comparison(
    summary_df: pd.DataFrame,
) -> dict:
    """
    Generate group comparison plots for each parameter in the summary dataframe.

    Parameters:
        summary_df (pd.DataFrame): The summary dataframe containing the data for comparison.
        log_params (dict): A dictionary to log any errors encountered during plotting.

    Returns:
        dict: A dictionary containing the generated group comparison plots for each parameter.
    """
    print('Generating group comparisons...')
    group_mean_parameter_figs = {}

    # get the parameters to compare
    parameters_to_compare = [column for column in summary_df.columns if 'Mean' in column]

    # generate and save figures for each parameter
    for param in parameters_to_compare:
        try:
            # Create a boxplot and swarmplot
            ax = sns.boxplot(x='Group Name', 
                            y=param, 
                            hue='Group Name',  # Assign 'Group Name' to 'hue'
                            data=summary_df, 
                            palette="Set2", 
                            showfliers=False,
                            )  
            ax = sns.swarmplot(x='Group Name', 
                            y=param, 
                            data=summary_df, 
                            color=".25")	
            ax.set_xticks(ax.get_xticks())  # Set the tick positions explicitly
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            fig = ax.get_figure()
            
            group_mean_parameter_figs[param] = fig
            plt.close(fig)

        except ValueError:
            logging.error(f'No data to compare for {param}')

    return group_mean_parameter_figs