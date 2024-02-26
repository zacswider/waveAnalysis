import os
import sys
import csv
import datetime
import numpy as np
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
            ax = sns.boxplot(x='Group Name', y=param, data=summary_df, palette = "Set2", showfliers = False)
            ax = sns.swarmplot(x='Group Name', y=param, data=summary_df, color=".25")	
            ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
            fig = ax.get_figure()
            fig.savefig(f'{group_save_path}/{param}.png')  # type: ignore
            plt.close(fig)
        except ValueError:
            log_params['Plotting errors'].append(f'No data to compare for {param}')

def group_name_error_check(
    file_names: list,
    group_names: list,
    log_params: dict
):
    # list of groups that matched to file names
    groups_found = np.unique([group for group in group_names for file in file_names if group in file]).tolist()

    # dictionary of file names and their corresponding group names
    uniqueDic = {file : [group for group in group_names if group in file] for file in file_names}

    for file_name, matching_groups in uniqueDic.items():
        # if a file doesn't have a group name, log it but still run the script
        if len(matching_groups) == 0:
            log_params["Group Matching Errors"].append(f'{file_name} was not matched to a group')

        # if a file has multiple groups names, raise error and exit the script
        elif len(matching_groups) > 1:
            print('****** ERROR ******',
                f'\n{file_name} matched to multiple groups: {matching_groups}',
                '\nPlease fix errors and try again.',
                '\n****** ERROR ******')
            sys.exit()

    # if a group was specified but not matched to a file name, raise error and exit the script
    if len(groups_found) != len(group_names):
        print("****** ERROR ******",
            "\nOne or more groups were not matched to file names",
            f"\nGroups specified: {group_names}",
            f"\nGroups found: {groups_found}",
            "\n****** ERROR ******")
        sys.exit()

def check_and_make_save_path(path=str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_plots(dict_of_plots: dict, save_path: str):
    for plot_name, plot in dict_of_plots.items():
        plot.savefig(f'{save_path}/{plot_name}.png')

def save_values_to_csv(
    ind_ccf_values: dict, 
    ind_ccf_val_path: str
):
    
    #save the indv CCF values for each bin to csv file
    for indv_ccfs_filename, measurements in ind_ccf_values.items():
        ind_ccf_path = os.path.join(ind_ccf_val_path, f'{indv_ccfs_filename}.csv')
        # Write measurements to CSV file
        with open(ind_ccf_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Ch1_Value', 'Ch2_Value', 'CCF_Value'])
            for time, ch1_val, ch2_val, ccf_val in measurements:
                writer.writerow([time, ch1_val, ch2_val, ccf_val])                
