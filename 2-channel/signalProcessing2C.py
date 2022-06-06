import os                                       
import sys 
import pathlib   
import timeit
import datetime
import numpy as np
import pandas as pd
from custom_gui import CustomGUI
from signal_processing_class import SignalProcessor

np.seterr(divide='ignore', invalid='ignore')

'''** GUI Window and sanity checks'''
# make GUI object and display the window
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

''' ** housekeeping functions ** '''
def make_log(directory, logParams):
    '''
    Convert dictionary of parameters to a log file and save it in the directory
    '''
    logPath = os.path.join(directory, "0_log.txt")                    # path to log file
    now = datetime.datetime.now()                                   # get current date and time
    logFile = open(logPath, "w")                                    # initiate text file
    logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")     # write current date and time
    for key, value in logParams.items():                            # for each key:value pair in the parameter dictionary...
        logFile.write('%s: %s\n' % (key, value))                    # write pair to new line
    logFile.close()                                                 # close the file

''' ** error catching for group names ** '''
# list of file names in specified directory
file_names = filelist = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

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

''' ** Main Workflow ** '''
# performance tracker
start = timeit.default_timer()
# empty list to fill with file stats
stats_list = []
# emtpy list to fill with column headers
col_headers = []
# create main save path
main_save_path = os.path.join(folder_path, "0_signalProcessing")
# create directory if it doesn't exist
if not os.path.exists(main_save_path):
    os.makedirs(main_save_path)

# empty list to fill with summary data for each file
summary_list = []
# column headers to use with summary data during conversion to dataframe
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
    
    # if file is not skipped, log it and continue
    log_params['Files Processed'].append(f'{file_name}')

    # empty dictionary to fill with summary statistics for the current file
    file_data_summary = {}

    # name without the extension
    name_wo_ext = file_name.rsplit(".",1)[0]

    # set save path for output for each image file
    boxSavePath = pathlib.Path(folder_path + "/0_signalProcessing/" + name_wo_ext)
    boxSavePath.mkdir(exist_ok=True, parents=True)
    
    # if user entered group name(s) into GUI, match the group for this file
    if group_names != ['']:                                      # if user entered group names to compare...
        group_name = [group for group in group_names if group in name_wo_ext][0]

    # calculate the number of boxes used for analysis
    num_boxes = processor.x_boxes * processor.y_boxes

    # calculate autocorrelation for each channel for each box
    acf_results = processor.calc_ACF(peak_thresh = acf_peak_thresh)

    # if possible, calculate the cross correlation between channels for each box
    if processor.num_channels == 2:
        ccf_results = processor.calc_CCF()

    # calculate the peak properties (width, max, min, amp, relAmp) for each channel for each box
    peak_properties = processor.calc_peaks()

    # consolidate results into a single dataframe
    # initial column names
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
    for box in range(num_boxes):
        # add box number to column names
        col_names.append(f"Box {box}")
    
    # initialize lists to fill with measurmemets for each box and summary statistics
    ch1_period_measurements = []
    ch1_width_measurements = []
    ch1_max_measurements = []
    ch1_min_measurements = []
    ch1_amp_measurements = []
    ch1_relAmp_measurements = []
    if processor.num_channels == 2:
        ch2_period_measurements = []
        ch2_width_measurements = []
        ch2_max_measurements = []
        ch2_min_measurements = []
        ch2_amp_measurements = []
        ch2_relAmp_measurements = []
        shift_measurements = []
    
    # summarize period measurements for each box
    for key, value in acf_results.items():
        if 'Ch1' in key:
            ch1_period_measurements.append(value[0])
        if processor.num_channels == 2 and 'Ch2' in key:
            ch2_period_measurements.append(value[0])
    for key, value in peak_properties.items():
        if 'Ch1' in key:
            ch1_width_measurements.append(value[0])
            ch1_max_measurements.append(value[1])
            ch1_min_measurements.append(value[2])
            ch1_amp_measurements.append(value[3])
            ch1_relAmp_measurements.append(value[4])
        if processor.num_channels == 2 and 'Ch2' in key:
            ch2_width_measurements.append(value[0])
            ch2_max_measurements.append(value[1])
            ch2_min_measurements.append(value[2])
            ch2_amp_measurements.append(value[3])
            ch2_relAmp_measurements.append(value[4])
    if processor.num_channels == 2:
        for value in ccf_results.values():
            shift_measurements.append(value[0])
         
    # calculate the number of boxes with no period for each channel
    ch1_pcnt_no_period = ((num_boxes-len(ch1_period_measurements))/num_boxes)*100
    if processor.num_channels == 2:
        ch2_pcnt_no_period = ((num_boxes-len(ch2_period_measurements))/num_boxes)*100

    # summarize statistics in file_data_summary
    file_data_summary['File Name'] = file_name
    file_data_summary['Num Boxes'] = num_boxes
    file_data_summary['Ch1 % Zero Boxes'] = ch1_pcnt_no_period
    if processor.num_channels == 2:
        file_data_summary['Ch2 % Zero Boxes'] = ch2_pcnt_no_period
    if group_names != ['']:
        file_data_summary['Group Name'] = group_name
    if processor.num_channels == 2:
        file_data_summary['Mean Signal Shift'] = np.mean(shift_measurements)
        file_data_summary['Median Signal Shift'] = np.median(shift_measurements)
        file_data_summary['StdDev Signal Shift'] = np.std(shift_measurements)
        file_data_summary['SEM Signal Shift'] = np.std(shift_measurements) / np.sqrt(len(shift_measurements))
    file_data_summary['Ch1 Mean Period'] = np.mean(ch1_period_measurements)
    file_data_summary['Ch1 Median Period'] = np.median(ch1_period_measurements)
    file_data_summary['Ch1 StdDev Period'] = np.std(ch1_period_measurements)
    file_data_summary['Ch1 SEM Period'] = np.std(ch1_period_measurements) / np.sqrt(len(ch1_period_measurements))
    file_data_summary['Ch1 Mean Width'] = np.mean(ch1_width_measurements)
    file_data_summary['Ch1 Median Width'] = np.median(ch1_width_measurements)
    file_data_summary['Ch1 StdDev Width'] = np.std(ch1_width_measurements)
    file_data_summary['Ch1 SEM Width'] = np.std(ch1_width_measurements) / np.sqrt(len(ch1_width_measurements))
    file_data_summary['Ch1 Mean Max'] = np.mean(ch1_max_measurements)
    file_data_summary['Ch1 Median Max'] = np.median(ch1_max_measurements)
    file_data_summary['Ch1 StdDev Max'] = np.std(ch1_max_measurements)
    file_data_summary['Ch1 SEM Max'] = np.std(ch1_max_measurements) / np.sqrt(len(ch1_max_measurements))
    file_data_summary['Ch1 Mean Min'] = np.mean(ch1_min_measurements)
    file_data_summary['Ch1 Median Min'] = np.median(ch1_min_measurements)
    file_data_summary['Ch1 StdDev Min'] = np.std(ch1_min_measurements)
    file_data_summary['Ch1 SEM Min'] = np.std(ch1_min_measurements) / np.sqrt(len(ch1_min_measurements))
    file_data_summary['Ch1 Mean Amp'] = np.mean(ch1_amp_measurements)
    file_data_summary['Ch1 Median Amp'] = np.median(ch1_amp_measurements)
    file_data_summary['Ch1 StdDev Amp'] = np.std(ch1_amp_measurements)
    file_data_summary['Ch1 SEM Amp'] = np.std(ch1_amp_measurements) / np.sqrt(len(ch1_amp_measurements))
    file_data_summary['Ch1 Mean RelAmp'] = np.mean(ch1_relAmp_measurements)
    file_data_summary['Ch1 Median RelAmp'] = np.median(ch1_relAmp_measurements)
    file_data_summary['Ch1 StdDev RelAmp'] = np.std(ch1_relAmp_measurements)
    file_data_summary['Ch1 SEM RelAmp'] = np.std(ch1_relAmp_measurements) / np.sqrt(len(ch1_relAmp_measurements))
    if processor.num_channels == 2:
        file_data_summary['Ch2 Mean Period'] = np.mean(ch2_period_measurements)
        file_data_summary['Ch2 Median Period'] = np.median(ch2_period_measurements)
        file_data_summary['Ch2 StdDev Period'] = np.std(ch2_period_measurements)
        file_data_summary['Ch2 SEM Period'] = np.std(ch2_period_measurements) / np.sqrt(len(ch2_period_measurements))
        file_data_summary['Ch2 Mean Width'] = np.mean(ch2_width_measurements)
        file_data_summary['Ch2 Median Width'] = np.median(ch2_width_measurements)
        file_data_summary['Ch2 StdDev Width'] = np.std(ch2_width_measurements)
        file_data_summary['Ch2 SEM Width'] = np.std(ch2_width_measurements) / np.sqrt(len(ch2_width_measurements))
        file_data_summary['Ch2 Mean Max'] = np.mean(ch2_max_measurements)
        file_data_summary['Ch2 Median Max'] = np.median(ch2_max_measurements)
        file_data_summary['Ch2 StdDev Max'] = np.std(ch2_max_measurements)
        file_data_summary['Ch2 SEM Max'] = np.std(ch2_max_measurements) / np.sqrt(len(ch2_max_measurements))
        file_data_summary['Ch2 Mean Min'] = np.mean(ch2_min_measurements)
        file_data_summary['Ch2 Median Min'] = np.median(ch2_min_measurements)
        file_data_summary['Ch2 StdDev Min'] = np.std(ch2_min_measurements)
        file_data_summary['Ch2 SEM Min'] = np.std(ch2_min_measurements) / np.sqrt(len(ch2_min_measurements))
        file_data_summary['Ch2 Mean Amp'] = np.mean(ch2_amp_measurements)
        file_data_summary['Ch2 Median Amp'] = np.median(ch2_amp_measurements)
        file_data_summary['Ch2 StdDev Amp'] = np.std(ch2_amp_measurements)
        file_data_summary['Ch2 SEM Amp'] = np.std(ch2_amp_measurements) / np.sqrt(len(ch2_amp_measurements))
        file_data_summary['Ch2 Mean RelAmp'] = np.mean(ch2_relAmp_measurements)
        file_data_summary['Ch2 Median RelAmp'] = np.median(ch2_relAmp_measurements)
        file_data_summary['Ch2 StdDev RelAmp'] = np.std(ch2_relAmp_measurements)
        file_data_summary['Ch2 SEM RelAmp'] = np.std(ch2_relAmp_measurements) / np.sqrt(len(ch2_relAmp_measurements))
    
    # function to summarize measurments statistics by appending them to the beginning of the measurement list
    def add_stats(measurements: list, measurement_name: str):
        '''
        Accepts a list of measurements. Calculates the mean, median, standard deviation, and SEM,
        and append them to the beginning of the list in that order. Finally, appends the name of
        the measurement of the beginning of the list.
        '''
        meas_mean = np.mean(measurements)
        meas_median = np.median(measurements)
        meas_std = np.std(measurements)
        meas_sem = meas_std / np.sqrt(len(measurements))
        measurements.insert(0, meas_mean)
        measurements.insert(1, meas_median)
        measurements.insert(2, meas_std)
        measurements.insert(3, meas_sem)
        measurements.insert(0, measurement_name)
        return measurements

    # insert Mean, Median, StdDev, and SEM into the beginning of each period list
    ch1_period_measurements = add_stats(ch1_period_measurements, "Ch1 Period")
    ch1_amp_measurements = add_stats(ch1_amp_measurements, "Ch1 Amplitude")
    ch1_width_measurements = add_stats(ch1_width_measurements, "Ch1 Width")
    ch1_max_measurements = add_stats(ch1_max_measurements, "Ch1 Max")
    ch1_min_measurements = add_stats(ch1_min_measurements, "Ch1 Min")
    ch1_relAmp_measurements = add_stats(ch1_relAmp_measurements, "Ch1 Relative Amplitude")

    if processor.num_channels == 2:
        ch2_period_measurements = add_stats(ch2_period_measurements, "Ch2 Period")
        shift_measurements = add_stats(shift_measurements, "Shift")
        ch2_amp_measurements = add_stats(ch2_amp_measurements, "Ch2 Amplitude")
        ch2_width_measurements = add_stats(ch2_width_measurements, "Ch2 Width")
        ch2_max_measurements = add_stats(ch2_max_measurements, "Ch2 Max")
        ch2_min_measurements = add_stats(ch2_min_measurements, "Ch2 Min")
        ch2_relAmp_measurements = add_stats(ch2_relAmp_measurements, "Ch2 Relative Amplitude")
    
    # create a subfolder within the main save path with the same name as the image file
    im_save_path = os.path.join(main_save_path, name_wo_ext)
    if not os.path.exists(im_save_path):
        os.makedirs(im_save_path)

    # save the summarized measurements as a .csv file in the main save path
    im_measurements = pd.DataFrame(columns = col_names, data = [ch1_period_measurements, 
                                                                ch2_period_measurements,
                                                                ch1_width_measurements,
                                                                ch2_width_measurements,
                                                                ch1_max_measurements,
                                                                ch2_max_measurements,
                                                                ch1_min_measurements,
                                                                ch2_min_measurements,
                                                                ch1_amp_measurements,
                                                                ch2_amp_measurements,
                                                                ch1_relAmp_measurements,
                                                                ch2_relAmp_measurements,
                                                                shift_measurements])

    im_measurements.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)

    # populate column headers list with keys from the measurements dictionary
    for key in file_data_summary.keys(): 
        if key not in col_headers: 
            col_headers.append(key) 
    
    # append summary data to the summary list
    summary_list.append(file_data_summary)

# convert summary_list to dataframe using the column headers
summary_df = pd.DataFrame(summary_list, columns = col_headers)
# save summary_df to main save path
summary_df.to_csv(f'{main_save_path}/summary.csv', index = False)

end = timeit.default_timer()
log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
# log parameters and errors
make_log(main_save_path, log_params)
