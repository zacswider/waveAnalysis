import os                                       
import sys 
import timeit
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from custom_gui import BaseGUI, RollingGUI
from signal_processing_class import SignalProcessor

np.seterr(divide='ignore', invalid='ignore')

'''** GUI Window and sanity checks'''
# make GUI object and display the window
gui = BaseGUI()
gui.mainloop()
# get GUI parameters
rolling = False
box_size = gui.box_size
folder_path = gui.folder_path
group_names = gui.group_names
acf_peak_thresh = gui.acf_peak_thresh
plot_ind_ACFs = gui.plot_ind_ACFs
plot_ind_CCFs = gui.plot_ind_CCFs
plot_ind_peaks = gui.plot_ind_peaks
# if rolling GUI specified, make rolling GUI object and display the window
if gui.roll:
    rolling = True
    gui = RollingGUI()
    gui.mainloop()
    # get GUI parameters
    box_size = gui.box_size
    folder_path = gui.folder_path
    acf_peak_thresh = gui.acf_peak_thresh
    plot_sf_ACFs = gui.plot_sf_ACFs
    plot_sf_CCFs = gui.plot_sf_CCFs
    plot_sf_peaks = gui.plot_sf_peaks
    subframe_size = gui.subframe_size
    subframe_roll = gui.subframe_roll

    group_names = ['']
    plot_ind_ACFs = False
    plot_ind_CCFs = False
    plot_ind_peaks = False

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
                "Files Not Processed" : [],
                'Plotting errors' : []
             } 
if rolling:
    log_params = {  "Box Size(px)" : box_size,
                    "Base Directory" : folder_path,
                    "ACF Peak Prominence" : acf_peak_thresh,
                    "Plot sub-movie ACFs" : plot_sf_ACFs,
                    "Plot movie CCFs" : plot_sf_CCFs,
                    "Plot movie Peaks" : plot_sf_peaks,
                    "Files Processed" : [],
                    "Files Not Processed" : [],
                    'Subframes Used' : []
             } 

''' ** housekeeping functions ** '''
def make_log(directory, logParams):
    '''
    Convert dictionary of parameters to a log file and save it in the directory
    '''
    now = datetime.datetime.now()
    logPath = os.path.join(directory, f"0_log-{now.strftime('%Y%m%d%H%M')}.txt")
    logFile = open(logPath, "w")                                    
    logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")     
    for key, value in logParams.items():                            
        logFile.write('%s: %s\n' % (key, value))                    
    logFile.close()                                                 

def plotComparisons(dataFrame: pd.DataFrame, dependent: str, independent = 'Group Name'):
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
now = datetime.datetime.now()
main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
# create directory if it doesn't exist
if not os.path.exists(main_save_path):
    os.makedirs(main_save_path)

# empty list to fill with summary data for each file
summary_list = []
# column headers to use with summary data during conversion to dataframe
col_headers = []

print('Processing files...')
with tqdm(total = len(file_names)) as pbar:
    for file_name in file_names: 
        if not rolling:
            processor = SignalProcessor(image_path = f'{folder_path}/{file_name}', box_size = box_size)
        if rolling:
            processor = SignalProcessor(image_path = f'{folder_path}/{file_name}', box_size = box_size, roll = rolling, roll_size = subframe_size, roll_by = subframe_roll)

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

        # name without the extension
        name_wo_ext = file_name.rsplit(".",1)[0]
        
        # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
        group_name = None
        if group_names != ['']:
            try:
                group_name = [group for group in group_names if group in name_wo_ext][0]
            except IndexError:
                pass

        # calculate the number of boxes used for analysis
        num_boxes = processor.x_boxes * processor.y_boxes

        # for rolling analyis, calculate the numbe of subframes used
        if rolling:
            num_subframes = processor.num_subframes
            log_params['Subframes Used'].append(num_subframes)

        # calculate autocorrelation for each channel for each box (for each subframe, if applicable)
        acf_results = processor.calc_ACF(peak_thresh = acf_peak_thresh)

        # if applicable, calculate the cross correlation between channels for each box
        if processor.num_channels == 2:
            ccf_results = processor.calc_CCF()

        # calculate the peak properties (width, max, min, amp, relAmp) for each channel for each box
        peak_properties = processor.calc_peaks()
        
        # create a subfolder within the main save path with the same name as the image file
        im_save_path = os.path.join(main_save_path, name_wo_ext)
        if not os.path.exists(im_save_path):
            os.makedirs(im_save_path)

        # for standard analysis, summarize the data for current image as dataframe, and save as .csv
        if not rolling:
            im_measurements_df, im_summary_dict = processor.summarize_results(file_name = file_name, group_name = group_name)
            im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)
        # for rolling analysis, summarize the data for each subframe as dataframe, and save as .csv
        if rolling:
            im_measurements_dfs, im_summary_df = processor.summarize_rolling_results(file_name = file_name)
            csv_save_path = os.path.join(im_save_path, 'rolling_measurements')
            if not os.path.exists(csv_save_path):
                os.makedirs(csv_save_path)
            for measurement_index, measurment in enumerate(im_measurements_dfs.values()):
                # save as csv
                measurment.to_csv(f'{csv_save_path}/{name_wo_ext}_subframe{measurement_index}_measurements.csv', index = False)
            # save summary dataframe as csv
            im_summary_df.to_csv(f'{im_save_path}/{name_wo_ext}_summary.csv', index = False)

        # populate column headers list with keys from the measurements dictionary
        if not rolling:
            for key in im_summary_dict.keys(): 
                if key not in col_headers: 
                    col_headers.append(key) 
        
            # append summary data to the summary list
            summary_list.append(im_summary_dict)

        # plot and save the population autocorrelation results for each channel
        if plot_ind_ACFs:
            acf_plots = processor.plot_mean_CF()
            ch1_acf_plot = acf_plots['Ch1 ACF']
            ch1_acf_plot.savefig(f'{im_save_path}/{name_wo_ext}_Ch1_ACF.png')
            if processor.num_channels == 2:
                ch2_acf_plot = acf_plots['Ch2 ACF']
                ch2_acf_plot.savefig(f'{im_save_path}/{name_wo_ext}_Ch2_ACF.png')
                ccf_plot = acf_plots['Mean CCF']
                ccf_plot.savefig(f'{im_save_path}/{name_wo_ext}_CCF.png')
        
        # make and save the summary plot for rolling data
        if rolling:
            summary_plots = processor.plot_rolling_summary()
            plot_save_path = os.path.join(im_save_path, 'summary_plots')
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)
            for title, plot in summary_plots.items():
                plot.savefig(f'{plot_save_path}/{name_wo_ext}_{title}.png')

        # plot and save the population peak properties for each channel
        if plot_ind_peaks:
            peak_plots = processor.plot_peak_props()
            ch1_peak_plot = peak_plots['Ch1']
            ch1_peak_plot.savefig(f'{im_save_path}/{name_wo_ext}_Ch1_Peaks.png')
            if processor.num_channels == 2:
                ch2_peak_plot = peak_plots['Ch2']
                ch2_peak_plot.savefig(f'{im_save_path}/{name_wo_ext}_Ch2_Peaks.png')
        pbar.update(1)


# convert summary_list to dataframe using the column headers and save to main save path
if not rolling:
    summary_df = pd.DataFrame(summary_list, columns = col_headers)
    summary_df.to_csv(f'{main_save_path}/summary.csv', index = False)

# if group names were entered into the gui, generate comparisons between each group
if group_names != ['']:
    print('Generating group comparisons...')
    # make a group comparisons save path in the main save directory
    group_save_path = os.path.join(main_save_path, "0_groupComparisons")
    if not os.path.exists(group_save_path):
        os.makedirs(group_save_path)
    
    # make a list of parameters to compare
    params_to_compare = ['Ch1 Mean Period', 
                         'Ch1 Mean Amp', 
                         'Ch1 Mean Width', 
                         'Ch1 Mean Max', 
                         'Ch1 Mean Min', 
                         'Ch1 Mean RelAmp']
    if processor.num_channels == 2:
        params_to_compare.extend(['Ch2 Mean Period', 
                                  'Ch2 Mean Amp', 
                                  'Ch2 Mean Width', 
                                  'Ch2 Mean Max', 
                                  'Ch2 Mean Min', 
                                  'Ch2 Mean RelAmp'])

    # generate and save figures for each parameter
    for param in params_to_compare:
        try:
            fig = plotComparisons(summary_df, param)
            fig.savefig(f'{group_save_path}/{param}.png')
            plt.close(fig)
        except ValueError:
            log_params['Plotting errors'].append(f'No data to compare for {param}')


end = timeit.default_timer()
log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
# log parameters and errors
make_log(main_save_path, log_params)
print('Done!')
