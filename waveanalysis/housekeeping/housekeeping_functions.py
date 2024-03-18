import os
import sys
import csv
import datetime
import numpy as np
from typing import Union, List, Tuple

def make_log(
    directory: str, 
    logParams: dict
) -> None:
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
    file_names: list[str],
    group_names: list[str],
    log_params: dict
) -> None:
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

def save_plots(
    dict_of_plots: dict, 
    save_path: str
) -> None:
    for plot_name, plot in dict_of_plots.items():
        plot.savefig(f'{save_path}/{plot_name}.png')

def match_group_to_file(
    name_wo_ext: str, 
    group_names: list[str]
) -> str:
    group_name = None
    if group_names != ['']:
        try:
            group_name = [group for group in group_names if group in name_wo_ext][0]
        except IndexError:
            pass

    return group_name     

def get_channel_combos(num_channels: int) -> list[list[int]]:
    channels = list(range(num_channels))
    channel_combos = []
    for i in range(num_channels):
        for j in channels[i+1:]:
            channel_combos.append([channels[i],j])

    return channel_combos

def threshold_check(
    threshold: float,
    log_params: dict
) -> dict:
    if threshold > 1:
        log_params["Errors"].append("The ACF peak prominence can not be greater than 1")
        log_params["Errors"].append("Set 'ACF peak prominence threshold' to a value between 0 and 1")
        log_params["Errors"].append("More realistically, a value between 0 and 0.5")
        return log_params

# TODO: move to save module
def save_values_to_csv(
    values: dict, 
    path: str,
):
    for filename, measurements in values.items():
        file_path = os.path.join(path, f'{filename}.csv')
        headers, data = determine_structure_and_values(measurements)
        write_to_csv(file_path, headers, data)

def determine_structure_and_values(measurements: Union[List[Tuple], List[List]]) -> Tuple[List[str], List[Tuple]]:
    # Check the structure of the measurements to determine headers and values
    first_entry = measurements[0]
    if len(first_entry) == 4:
        # Individual CCFs
        headers = ['Time', 'Ch1_Value', 'Ch2_Value', 'CCF_Value']
    elif len(first_entry) == 3:
        # Mean CCFs
        headers = ['Time', 'Mean', 'StDev']
    else:
        raise ValueError("Unsupported measurements format")

    return headers, measurements

def write_to_csv(file_path: str, headers: List[str], data: List[Tuple]):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)