import os                                       
import sys 
import pathlib   
import timeit
import numpy as np
from custom_gui import CustomGUI
from signal_processing_class import SignalProcessor

np.seterr(divide='ignore', invalid='ignore')

'''** GUI Window and sanity checks'''
# make GUI object
gui = CustomGUI()
gui.mainloop()

# identify and report errors in GUI input
errors = []
if gui.acf_peak_thresh > 1 :
    errors.append("The ACF peak prominence can not be greater than 1",
                  ", set 'ACF peak prominence threshold' to a value between 0 and 1.",
                  "More realistically, a value between 0 and 0.5")
if len(gui.folder_path) < 1 :
    errors.append("You didn't enter a directory to analyze")

if len(errors) >= 1 :
    print("Error Log:")
    for count, error in enumerate(errors):
        print(count,":", error)
    sys.exit("Please fix errors and try again.") 

# get GUI parameters
box_size = gui.box_size
folder_path = gui.folder_path
group_names = gui.group_names
acf_peak_thresh = gui.acf_peak_thresh
plot_ind_ACFs = gui.plot_ind_ACFs
plot_ind_CCFs = gui.plot_ind_CCFs
plot_ind_peaks = gui.plot_ind_peaks

#make dictionary of parameters for log file use
log_params = {  "Box Size(px)" : box_size,
                "Base Directory" : folder_path,
                "ACF Peak Prominence" : acf_peak_thresh,
                "Group Names" : group_names,
                "Plot Individual ACFs" : plot_ind_ACFs,
                "Plot Individual CCFs" : plot_ind_CCFs,
                "Plot Individual Peaks" : plot_ind_peaks,
                "Group Matching Errors" : [],
                "Files Processed" : [],
                "Files Not Processed" : []
             }

''' ** error catching for group names ** '''
# list of file names in specified directory
file_names = filelist = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

# list of groups that matched to file names
groups_found = np.unique([group for group in group_names for file in file_names if group in file]).tolist()

# dictionary of file names and their corresponding group names
uniqueDic = {file : [group for group in group_names if group in file] for file in file_names}

for file_name, group_names in uniqueDic.items():
    # if a file doesn't have a group name, log it but still run the script
    if len(group_names) == 0:
        log_params["Group Matching Errors"].append(f'{file_name} was not matched to a group')

    # if a file has multiple groups names, raise error and exit the script
    elif len(group_names) > 1:
        print('****** ERROR ******',
             f'\n{file_name} matched to multiple groups: {group_names}',
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

''' ** Main Workflow ** '''
# performance tracker
start = timeit.default_timer()
# empty list to fill with file stats
stats_list = []
# emtpy list to fill with column headers
col_headers = []

for file_name in file_names: 
    print(f"Starting to work on {file_name}")
    processor = SignalProcessor(image_path = f'{folder_path}/{file_name}', box_size = box_size)

    # log error and skip image if frames < 2 or channels >2
    if processor.num_frames < 2:
        print(f"****** ERROR ******",
              f"\n{file_name} has less than 2 frames",
              "\n****** ERROR ******")
        log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
        continue
    if processor.num_channels > 2:
        print(f"****** ERROR ******",
              f"\n{file_name} has more than 2 channels",
              "\n****** ERROR ******")
        log_params['Files Not Processed'].append(f'{file_name} has more than 2 channels')
        continue
    
    # name without the extension
    name_wo_ext = file_name.rsplit(".",1)[0]

    # set save path for output for each image file
    boxSavePath = pathlib.Path(folder_path + "/0_signalProcessing/" + name_wo_ext)
    boxSavePath.mkdir(exist_ok=True, parents=True)
    
    # if user entered group name(s) into GUI, match the group for this file
    if group_names != ['']:                                      # if user entered group names to compare...
        group_name = [group for group in group_names if group in name_wo_ext][0]


