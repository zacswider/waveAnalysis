import os
import sys
import datetime
import numpy as np

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
    
def check_if_wave_tracks_created(
    wave_tracks,
    log_params: dict,
    file_name: str
) -> dict:
    if len(wave_tracks) == 0:
        log_params["Errors"].append(f'No wave tracks were found for {file_name}')

def check_wave_track_coords(
    wave_tracks,
    log_params: dict,
    file_name: str,
    num_columns: int,
    num_frames: int
) -> dict:
    for track in wave_tracks:
        x1, x2 = track[0][1], track[1][1]
        y1, y2 = track[0][0], track[1][0]
        if x1 < 0 or x1 >= num_columns or x2 < 0 or x2 >= num_columns or y1 < 0 or y1 >= num_frames or y2 < 0 or y2 >= num_frames:
            print(f"****** WARNING ******",
                f"\nAll lines are not drawn within the image for {file_name}",
                "\n****** WARNING ******")
            log_params['Errors'].append(f'All lines are not drawn within the image for {file_name}')

def check_frame_interval(
    frame_interval: float,
    log_params: dict,
    file_name: str
) -> dict:
    if frame_interval == None or frame_interval == 0 or frame_interval == 1:
        print(f"****** WARNING ******",
            f"\n{file_name} frame interval is not provided or 0 or 1. Ensure this is the correct value. All calculations will be done assuming a frame interval of 1.",
            "\n****** WARNING ******")
        log_params['Errors'].append(f'{file_name} frame interval is not provided is 0 or 1. Ensure this is the correct value. All calculations will be done assuming a frame interval of 1.')

        frame_interval = 1