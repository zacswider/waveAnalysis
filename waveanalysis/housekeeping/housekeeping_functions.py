import os
import sys
import logging
import datetime
import numpy as np

def make_log(
    directory: str, 
    logParams: dict
) -> None:
    """
    Creates a log file with the current timestamp and writes the log parameters to it.

    Args:
        directory (str): The directory where the log file will be created.
        logParams (dict): A dictionary containing the log parameters.
    """
    now = datetime.datetime.now()
    logPath = os.path.join(directory, f"!log-{now.strftime('%Y%m%d%H%M')}.txt")
    logFile = open(logPath, "w")                                    
    logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")     
    for key, value in logParams.items():                            
        logFile.write('%s: %s\n' % (key, value))                    
    logFile.close()

def group_name_error_check(
    file_names: list[str],
    group_names: list[str]
) -> None:
    """
    Check for errors in matching file names to group names.

    Args:
        file_names (list[str]): List of file names.
        group_names (list[str]): List of group names.
        log_params (dict): Dictionary to store log parameters.

    Raises:
        SystemExit: If a file has multiple group names or if a group was specified but not matched to a file name.
    """
    
    # list of groups that matched to file names
    groups_found = np.unique([group for group in group_names for file in file_names if group in file]).tolist()

    # dictionary of file names and their corresponding group names
    uniqueDic = {file : [group for group in group_names if group in file] for file in file_names}

    for file_name, matching_groups in uniqueDic.items():
        # if a file doesn't have a group name, log it but still run the script
        if len(matching_groups) == 0:
            logging.error('%s was not matched to a group', file_name)

        # if a file has multiple groups names, raise error and exit the script
        elif len(matching_groups) > 1:
            logging.error('%s matched to multiple groups: %s', file_name, matching_groups)
            sys.exit()

    # if a group was specified but not matched to a file name, raise error and exit the script
    if len(groups_found) != len(group_names):
        logging.error('One or more groups were not matched to file names.Groups specified: %s. Groups found: %s', group_names, groups_found)
        sys.exit()

def save_plots(
    dict_of_plots: dict, 
    save_path: str
) -> None:
    '''
    Save all plots in a dictionary to a specified path
    '''
    for plot_name, plot in dict_of_plots.items():
        plot.savefig(f'{save_path}/{plot_name}.png')

def match_group_to_file(
    name_wo_ext: str, 
    group_names: list[str]
) -> str:
    '''
    Match a group name to a file name
    '''
    group_name = None # default value
    if group_names != ['']:
        try:
            group_name = [group for group in group_names if group in name_wo_ext][0]
        except IndexError:
            pass

    return group_name     

def get_channel_combos(num_channels: int) -> list[list[int]]:
    '''
    Get all possible combinations of channels for cross-correlation
    '''
    channels = list(range(num_channels))
    channel_combos = []
    for i in range(num_channels):
        for j in channels[i+1:]:
            channel_combos.append([channels[i],j])

    return channel_combos

def threshold_check(
    threshold: float,
) -> None:
    '''
    Check if the ACF peak prominence threshold is greater than 1
    '''
    if threshold > 1:
        logging.error('The ACF peak prominence can not be greater than 1')
        logging.error('Set "ACF peak prominence threshold" to a value between 0 and 1')
        logging.error('More realistically, a value between 0 and 0.5')
        
def check_if_wave_tracks_created(
    wave_tracks,
    file_name: str
) -> None:
    '''
    Check if wave tracks were created
    '''
    if len(wave_tracks) == 0:
        logging.error('No wave tracks were created for %s', file_name)

def check_wave_track_coords(
    wave_tracks,
    file_name: str,
    num_columns: int,
    num_frames: int
) -> None:
    """
    Check if the coordinates of wave tracks are within the image boundaries.

    Args:
        wave_tracks (list): List of wave tracks.
        log_params (dict): Dictionary to store log parameters.
        file_name (str): Name of the file being processed.
        num_columns (int): Number of columns in the image.
        num_frames (int): Number of frames in the image.
    """
    for track in wave_tracks:
        # get the x and y coordinates of the wave track
        x1, x2 = track[0][1], track[1][1]
        y1, y2 = track[0][0], track[1][0]
        # check if the coordinates are within the image boundaries
        if x1 < 0 or x1 >= num_columns or x2 < 0 or x2 >= num_columns or y1 < 0 or y1 >= num_frames or y2 < 0 or y2 >= num_frames:
            logging.error('Wave track coordinates are not within the image boundaries for %s', file_name)
            
def check_frame_interval(
    frame_interval: float,
    file_name: str
) -> float:
    """
    Check the validity of the frame interval and provide a default value if necessary.

    Args:
        frame_interval (float): The frame interval to be checked.
        log_params (dict): A dictionary containing log parameters.
        file_name (str): The name of the file being processed.

    Returns:
        float: The validated frame interval.
    """
    if frame_interval == None or frame_interval == 0 or frame_interval == 1 or np.isnan(frame_interval):
        logging.warning('%s frame interval is not provided or 0 or 1. Ensure this is the correct value. All calculations will be done assuming a frame interval of 1.', file_name)

        # set frame interval to 1 if it is not provided or 0
        frame_interval = 1
    
    return frame_interval

# make dictionary of parameters for log file use
    log_params = {  "Box Size(px)" : box_size,
                    "Box Shift(px)" : bin_shift,
                    "Base Directory" : folder_path,
                    "ACF Peak Prominence" : acf_peak_thresh,
                    "Group Names" : group_names,
                    "Plot Summary ACFs" : plot_summary_ACFs,
                    "Plot Summary CCFs" : plot_summary_CCFs,
                    "Plot Summary Peaks" : plot_summary_peaks,
                    "Plot Individual ACFs" : plot_indv_ACFs,
                    "Plot Individual CCFs" : plot_indv_CCFs,
                    "Plot Individual Peaks" : plot_indv_peaks,
                    'Calc Wave Speeds': False,
                    'Plot Wave Speeds': False,
                    "Files Processed" : [],
                    "Files Not Processed" : [],
                    "Errors" : [],
                    'Frame Interval': [],
                    'Pixel Size': [],
                } 
    
    if analysis_type == 'kymograph':
        log_params = {  "Line width": line_width,
                        "Line Shift(px)": bin_shift,
                        "Base Directory": folder_path,
                        "ACF Peak Prominence" : acf_peak_thresh,
                        "Group Names" : group_names,
                        "Plot Summary ACFs": plot_summary_ACFs,
                        "Plot Summary CCFs": plot_summary_CCFs,
                        "Plot Summary Peaks": plot_summary_peaks,
                        "Plot Individual ACFs": plot_indv_ACFs,
                        "Plot Individual CCFs": plot_indv_CCFs,
                        "Plot Individual Peaks": plot_indv_peaks,  
                        'Calc Wave Speeds': calc_wave_speeds,
                        'Plot Wave Speeds': True,
                        "Files Processed": [],
                        "Files Not Processed": [],
                        "Errors" : [],
                        'Frame Interval': [],
                        'Pixel Size': [],
                }