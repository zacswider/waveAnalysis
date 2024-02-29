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

def save_plots(dict_of_plots: dict, save_path: str):
    for plot_name, plot in dict_of_plots.items():
        plot.savefig(f'{save_path}/{plot_name}.png')

def match_group_to_file(
    name_wo_ext: str,
    group_names: list
):
    group_name = None
    if group_names != ['']:
        try:
            group_name = [group for group in group_names if group in name_wo_ext][0]
        except IndexError:
            pass

    return group_name

# TODO: move this to the saving module
def save_values_to_csv(
    values: dict, 
    path: str,
    indv_ccfs_bool: bool = False
):
    
    # TODO: figure out a way so that the code is not hard coded to the indv vs mean CCFs

    #save the indv CCF values for each bin to csv file
    for filename, measurements in values.items():
        path = os.path.join(path, f'{filename}.csv')
        # Write measurements to CSV file
        with open(path, 'w', newline='') as csvfile:
            if indv_ccfs_bool:
                writer = csv.writer(csvfile)
                writer.writerow(['Time', 'Ch1_Value', 'Ch2_Value', 'CCF_Value'])
                for time, ch1_val, ch2_val, ccf_val in measurements:
                    writer.writerow([time, ch1_val, ch2_val, ccf_val])                
            else:
                writer = csv.writer(csvfile)
                writer.writerow(['Time', 'Mean', 'StDev'])
                for time, mean, stdev in measurements:
                    writer.writerow([time, mean, stdev])         