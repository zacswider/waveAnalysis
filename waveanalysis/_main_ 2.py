import os                                       
import sys 
import csv
import timeit
import pathlib
import datetime
import tifffile
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from waveanalysismods.customgui_wave_analysis import BaseGUI, RollingGUI, KymographGUI
from waveanalysismods.processor_wave_analysis import TotalSignalProcessor, RollingSignalProcessor, KymographSignalProcessor

# set the behavior for two types of errors: divide-by-zero and invalid arithmetic operations 
np.seterr(divide='ignore', invalid='ignore')

# set this to zero so that warning does not pop up
plt.rcParams['figure.max_open_warning'] = 0

def convert_kymograph_images(directory):
    """
    Convert all TIF files in the specified directory to standardized numpy arrays.

    Args:
        directory (str or pathlib.Path): The path to the directory containing the TIF files.

    Returns:
        dict: A dictionary where the keys are the file names and the values are the numpy arrays of the images.
    """
    input_path = pathlib.Path(directory)
    images = {}

    for file_path in input_path.glob('*.tif'):
        try:
            # Load the TIFF file into a numpy array
            image = tifffile.imread(file_path)

            # standardize image dimensions
            with tifffile.TiffFile(file_path) as tif_file:
                metadata = tif_file.imagej_metadata
            num_channels = metadata.get('channels', 1)
            image = image.reshape(num_channels, 
                                    image.shape[-2],  # rows
                                    image.shape[-1])  # cols
            
            images[file_path.name] = image
                        
        except tifffile.TiffFileError:
            print(f"Warning: Skipping '{file_path.name}', not a valid TIF file.")

    # Sort the dictionary keys alphabetically
    images = {key: images[key] for key in sorted(images)}

    return images   

####################################################################################################################################
####################################################################################################################################

def main():
    # make GUI object and display the window
    gui = BaseGUI()
    gui.mainloop()
    # get GUI parameters
    rolling = False
    kymograph_analysis = False
    box_size = gui.box_size
    box_shift = gui.box_shift
    folder_path = gui.folder_path
    group_names = gui.group_names
    acf_peak_thresh = gui.acf_peak_thresh
    plot_summary_ACFs = gui.plot_summary_ACFs
    plot_summary_CCFs = gui.plot_summary_CCFs
    plot_summary_peaks = gui.plot_summary_peaks
    plot_ind_ACFs = gui.plot_ind_ACFs
    plot_ind_CCFs = gui.plot_ind_CCFs
    plot_ind_peaks = gui.plot_ind_peaks
    save_ind_CCF_values = gui.save_ind_CCF_values

    # if rolling GUI specified, make rolling GUI object and display the window
    if gui.roll:
        rolling = True
        gui = RollingGUI()
        gui.mainloop()
        # get GUI parameters
        box_size = gui.box_size
        box_shift = gui.box_shift
        folder_path = gui.folder_path
        acf_peak_thresh = gui.acf_peak_thresh
        plot_sf_ACFs = gui.plot_sf_ACFs
        plot_sf_CCFs = gui.plot_sf_CCFs
        plot_sf_peaks = gui.plot_sf_peaks
        subframe_size = gui.subframe_size
        subframe_roll = gui.subframe_roll

        group_names = ['']
        plot_summary_ACFs = False
        plot_summary_CCFs = False
        plot_summary_peaks = False
    
    if gui.kymograph:
        # make GUI object and display the window
        kymograph_analysis = True
        gui = KymographGUI()
        gui.mainloop()
        folder_path = gui.folder_path
        plot_mean_CCFs = gui.plot_summary_CCFs
        plot_mean_peaks = gui.plot_summary_peaks
        plot_mean_acfs = gui.plot_summary_ACFs
        plot_ind_CCFs = gui.plot_ind_CCFs
        plot_ind_peaks = gui.plot_ind_peaks
        plot_ind_acfs = gui.plot_ind_ACFs
        fast_process = gui.fast_process
        line_width = gui.line_width
        group_names = gui.group_names

        acf_peak_thresh = gui.acf_peak_thresh

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
                    "Box Shift(px)" : box_shift,
                    "Base Directory" : folder_path,
                    "ACF Peak Prominence" : acf_peak_thresh,
                    "Group Names" : group_names,
                    "Plot Summary ACFs" : plot_summary_ACFs,
                    "Plot Summary CCFs" : plot_summary_CCFs,
                    "Plot Summary Peaks" : plot_summary_peaks,
                    "Plot Individual ACFs" : plot_ind_ACFs,
                    "Plot Individual CCFs" : plot_ind_CCFs,
                    "Save Individual CCF values" : save_ind_CCF_values,
                    "Plot Individual Peaks" : plot_ind_peaks,
                    "Group Matching Errors" : [],
                    "Files Processed" : [],
                    "Files Not Processed" : [],
                    'Plotting errors' : []
                } 
    if rolling:
        log_params = {  "Box Size(px)" : box_size,
                        "Box Shift(px)" : box_shift,
                        "Base Directory" : folder_path,
                        "ACF Peak Prominence" : acf_peak_thresh,
                        "Plot sub-movie ACFs" : plot_sf_ACFs,
                        "Plot movie CCFs" : plot_sf_CCFs,
                        "Plot movie Peaks" : plot_sf_peaks,
                        "Files Processed" : [],
                        "Files Not Processed" : [],
                        'Submovies Used' : []
                } 
    if kymograph_analysis:
        log_params = {"Base Directory": folder_path,
                  "Plot Summary ACFs": plot_mean_acfs,
                "Plot Summary CCFs": plot_mean_CCFs,
                "Plot Summary Peaks": plot_mean_peaks,
                "Plot Individual CCF": plot_mean_acfs,
                "Plot Individual CCFs": plot_ind_CCFs,
                "Plot Individual Peaks": plot_ind_peaks,  
                "Line width": line_width,
                "Group Names" : group_names,
                "Files Processed": [],
                "Files Not Processed": [],
                'Plotting errors': [],
                "Group Matching Errors" : [],
                "Multiprocessing": fast_process
                }

    def make_log(directory, logParams):
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

    def plotComparisons(dataFrame: pd.DataFrame, dependent: str, independent = 'Group Name'):
        '''
        This func accepts a dataframe, the name of a dependent variable, and the name of an
        independent variable (by default, set to Group Name). It returns a figure object showing
        a box and scatter plot of the dependent variable grouped by the independent variable.
        '''
        plt.style.use('dark_background')

        ax = sns.boxplot(x=independent, y=dependent, data=dataFrame, palette = "Set2", showfliers = False)
        ax = sns.swarmplot(x=independent, y=dependent, data=dataFrame, color=".25")	
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        fig = ax.get_figure()
        return fig
    
    def calc_anova(dataFrame: pd.DataFrame, dependent: str, independent = 'Group Name'):
        """
        Calculate ANOVA and pairwise comparisons.

        Parameters:
            dataFrame: Input DataFrame containing the data.
            dependent: Column name representing the dependent variable.
            independent: Column name representing the independent variable (default: 'Group Name').

        Returns:
            Tuple containing the ANOVA table and pairwise comparisons.
        """
        new_df = dataFrame[[independent, dependent]].copy()
        new_df = new_df.rename(columns={dependent: 'parameter', independent: 'group'})
        
        model = ols(formula='parameter ~ group', data=new_df).fit()
        anova_table = sm.stats.anova_lm(model)
        pairwise_comparisons = sm.stats.multicomp.pairwise_tukeyhsd(new_df['parameter'], new_df['group'])
        
        return anova_table, pairwise_comparisons
    
    # list of file names in specified directory
    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

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

    # performance tracker
    start = timeit.default_timer()

    # create main save path
    now = datetime.datetime.now()
    main_save_path = os.path.join(folder_path, f"!signalProcessing-{now.strftime('%Y%m%d%H%M')}")
    
    # create directory if it doesn't exist
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    # empty list to fill with summary data for each file
    summary_list = []
    # column headers to use with summary data during conversion to dataframe
    col_headers = []

    print('Processing files...')

    if rolling == False and kymograph_analysis == False:
        with tqdm(total = len(file_names)) as pbar:
            pbar.set_description('Files processed:')
            for file_name in file_names: 
                print('******'*10)
                print(f'Processing {file_name}...')
                processor = TotalSignalProcessor(image_path = f'{folder_path}/{file_name}', kern = box_size, step = box_shift)

                # log error and skip image if frames < 2 
                if processor.num_frames < 2:
                    print(f"****** ERROR ******",
                        f"\n{file_name} has less than 2 frames",
                        "\n****** ERROR ******")
                    log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
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

                # calculate the population signal properties
                processor.calc_ACF(peak_thresh = acf_peak_thresh)
                processor.calc_peak_props()
                if processor.num_channels > 1:
                    processor.calc_CCF()
                
                # create a subfolder within the main save path with the same name as the image file
                im_save_path = os.path.join(main_save_path, name_wo_ext)
                if not os.path.exists(im_save_path):
                    os.makedirs(im_save_path)

                # plot and save the population autocorrelation, crosscorrelation, and peak properties for each channel
                if plot_summary_ACFs:
                    summ_acf_plots = processor.plot_mean_ACF()
                    for plot_name, plot in summ_acf_plots.items():
                        plot.savefig(f'{im_save_path}/{plot_name}.png')
                if plot_summary_CCFs:
                    summ_ccf_plots, mean_ccf_values = processor.plot_mean_CCF()
                    for plot_name, plot in summ_ccf_plots.items():
                        plot.savefig(f'{im_save_path}/{plot_name}.png')
                    for csv_filename, CCF_values in mean_ccf_values.items():
                        with open(os.path.join(im_save_path, csv_filename), 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(['Time', 'CCF_Value', 'STD'])
                            for time, ccf_val, arr_std in CCF_values:
                                writer.writerow([time, ccf_val, arr_std])
                                
                if plot_summary_peaks:
                    summ_peak_plots = processor.plot_mean_peak_props()
                    for plot_name, plot in summ_peak_plots.items():
                        plot.savefig(f'{im_save_path}/{plot_name}.png')
                
                if plot_ind_peaks:
                    ind_peak_plots = processor.plot_ind_peak_props()
                    ind_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                    if not os.path.exists(ind_peak_path):
                        os.makedirs(ind_peak_path)
                    for plot_name, plot in ind_peak_plots.items():
                        plot.savefig(f'{ind_peak_path}/{plot_name}.png')

                if plot_ind_ACFs:
                    ind_acfs_plots = processor.plot_ind_acfs()
                    ind_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                    if not os.path.exists(ind_acf_path):
                        os.makedirs(ind_acf_path)
                    for plot_name, plot in ind_acfs_plots.items():
                        plot.savefig(f'{ind_acf_path}/{plot_name}.png')

                if plot_ind_CCFs:
                    if processor.num_channels == 1:
                            log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                    ind_ccf_plots = processor.plot_ind_ccfs()
                    ind_ccf_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                    if not os.path.exists(ind_ccf_path):
                        os.makedirs(ind_ccf_path)
                    for plot_name, plot in ind_ccf_plots.items():
                        plot.savefig(f'{ind_ccf_path}/{plot_name}.png')

                if save_ind_CCF_values:
                    if processor.num_channels == 1:
                        log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                    ind_ccf_val_path = os.path.join(im_save_path, 'Individual_CCF_values')
                    if not os.path.exists(ind_ccf_val_path):
                        os.makedirs(ind_ccf_val_path)
                    processor.save_ind_ccf_values(save_folder=ind_ccf_val_path)

                # Summarize the data for current image as dataframe, and save as .csv
                im_measurements_df = processor.organize_measurements()
                im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)

                # generate summary data for current image
                im_summary_dict = processor.summarize_image(file_name = file_name, group_name = group_name)

                # populate column headers list with keys from the measurements dictionary
                for key in im_summary_dict.keys(): 
                    if key not in col_headers: 
                        col_headers.append(key) 
            
                # append summary data to the summary list
                summary_list.append(im_summary_dict)

                # useless progress bar to force completion of previous bars
                with tqdm(total = 10, miniters = 1) as dummy_pbar:
                    dummy_pbar.set_description('cleanup:')
                    for i in range(10):
                        dummy_pbar.update(1)

                pbar.update(1)

            # create dataframe from summary list    
            summary_df = pd.DataFrame(summary_list, columns = col_headers)

            # save the summary csv file
            summary_df = summary_df.sort_values('File Name', ascending=True)
            summary_df.to_csv(f"{main_save_path}/!{now.strftime('%Y%m%d%H%M')}_summary.csv", index = False)

            # if group names were entered into the gui, generate comparisons between each group
            if group_names != ['']:
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
                anova_path = os.path.join(main_save_path, 'pairwise_comparisons.txt')
                with open(anova_path, 'w') as file:
                    for param in params_to_compare:
                        try:
                            if param != 'Ch 1 Mean Shift' and param != 'Ch 2 Mean Shift' and param != 'Ch 3 Mean Shift' and param != 'Ch 4 Mean Shift':
                                anova_table, pairwise_comparisons = calc_anova(summary_df, param)
                                file.write(f'{param}:\n{str(pairwise_comparisons)}\n{str(anova_table)}\n\n')
                            fig = plotComparisons(summary_df, param)
                            fig.savefig(f'{group_save_path}/{param}.png')
                            plt.close(fig)
                        except ValueError:
                            log_params['Plotting errors'].append(f'No data to compare for {param}')
                
                # save the means for the attributes to make them easier to work with in prism
                processor.save_means_to_csv(main_save_path, group_names, summary_df)

            end = timeit.default_timer()
            log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
            # log parameters and errors
            make_log(main_save_path, log_params)
            print('Done!')

    if rolling:
        with tqdm(total = len(file_names)) as pbar:
            for file_name in file_names: 
                print('******'*10)
                print(f'Processing {file_name}...')
                processor = RollingSignalProcessor(image_path = f'{folder_path}/{file_name}', kern = box_size, step = box_shift, roll_size = subframe_size, roll_by = subframe_roll)

                # log error and skip image if frames < 2 
                if processor.num_frames < 2:
                    print(f"****** ERROR ******",
                        f"\n{file_name} has less than 2 frames",
                        "\n****** ERROR ******")
                    log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                    continue
            
                # if file is not skipped, log it and continue
                log_params['Files Processed'].append(f'{file_name}')

                # name without the extension
                name_wo_ext = file_name.rsplit(".",1)[0]

                # calculate the number of subframes used
                num_submovies = processor.num_submovies
                log_params['Submovies Used'].append(num_submovies)

                # calculate the population signal properties
                processor.calc_ACF(peak_thresh = acf_peak_thresh)
                processor.calc_peak_props()
                if processor.num_channels > 1:
                    processor.calc_CCF()
            
                # create a subfolder within the main save path with the same name as the image file
                im_save_path = os.path.join(main_save_path, name_wo_ext)
                if not os.path.exists(im_save_path):
                    os.makedirs(im_save_path)

                # summarize the data for each subframe as individual dataframes, and save as .csv
                submovie_meas_list = processor.get_submovie_measurements()
                csv_save_path = os.path.join(im_save_path, 'rolling_measurements')
                if not os.path.exists(csv_save_path):
                    os.makedirs(csv_save_path)
                for measurement_index, submovie_meas_df in enumerate(submovie_meas_list):
                    submovie_meas_df.to_csv(f'{csv_save_path}/{name_wo_ext}_subframe{measurement_index}_measurements.csv', index = False)
                
                # summarize the data for each subframe as a single dataframe, and save as .csv
                summary_df = processor.summarize_file()
                summary_df.to_csv(f'{im_save_path}/{name_wo_ext}_summary.csv', index = False)

                # make and save the summary plot for rolling data
                summary_plots = processor.plot_rolling_summary()
                plot_save_path = os.path.join(im_save_path, 'summary_plots')
                if not os.path.exists(plot_save_path):
                    os.makedirs(plot_save_path)
                for title, plot in summary_plots.items():
                    plot.savefig(f'{plot_save_path}/{name_wo_ext}_{title}.png')
                pbar.update(1)

    if kymograph_analysis:
        # create a dictionary of the filename and corresponding images as mp arrays. 
        all_images = convert_kymograph_images(folder_path)
        # processing movies
        with tqdm(total=len(file_names)) as pbar:
            pbar.set_description('Files processed:')
            for file_name in file_names:
                print('******'*10)
                print(f'Processing {file_name}...')

                # name without the extension
                name_wo_ext = file_name.rsplit(".", 1)[0]

                # create a subfolder within the main save path with the same name as the image file
                im_save_path = os.path.join(main_save_path, name_wo_ext)
                if not os.path.exists(im_save_path):
                    os.makedirs(im_save_path)

                processor = KymographSignalProcessor(filename=file_name, 
                            im_save_path=im_save_path,
                            img=all_images[file_name],
                            line_width=line_width
                            )
            
                # log error and skip image if frames < 2 
                if processor.num_cols < 2:
                    print(f"****** ERROR ******",
                        f"\n{file_name} has less than 2 frames",
                        "\n****** ERROR ******")
                    log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                    continue

                # if file is not skipped, log it and continue
                log_params['Files Processed'].append(f'{file_name}')
                
                # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
                group_name = None
                if group_names != ['']:
                    try:
                        group_name = [group for group in group_names if group in name_wo_ext][0]
                    except IndexError:
                        pass

                # if file is not skipped, log it and continue
                log_params['Files Processed'].append(f'{file_name}')

                # calculate the population signal properties
                processor.calc_ind_peak_props()
                processor.calc_indv_ACF()
                if processor.num_channels > 1:
                    processor.calc_indv_CCFs()

                # Plot the following parameters if selected
                if fast_process:
                    if plot_ind_peaks:
                        ind_peak_plots = processor.plot_ind_peak_props()
                        processor.save_plots(plots=ind_peak_plots, plot_dir=os.path.join(im_save_path, 'Individual_peak_plots'))

                    if plot_ind_CCFs:
                        if processor.num_channels == 1:
                                log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                        ind_ccf_plots = processor.plot_ind_ccfs()
                        processor.save_plots(plots=ind_ccf_plots, plot_dir=os.path.join(im_save_path, 'Individual_CCF_plots'))

                    if plot_ind_acfs:
                        ind_acfs_plots = processor.plot_ind_acfs()
                        processor.save_plots(plots=ind_acfs_plots, plot_dir=os.path.join(im_save_path, 'Individual_ACF_plots'))
                
                else:
                    if plot_ind_peaks:
                        ind_peak_plots = processor.plot_ind_peak_props()
                        ind_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                        if not os.path.exists(ind_peak_path):
                            os.makedirs(ind_peak_path)
                        for plot_name, plot in ind_peak_plots.items():
                            plot.savefig(f'{ind_peak_path}/{plot_name}.png')

                    if plot_ind_CCFs:
                        if processor.num_channels == 1:
                                log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                        ind_ccf_plots = processor.plot_ind_ccfs()
                        ind_ccf_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                        if not os.path.exists(ind_ccf_path):
                            os.makedirs(ind_ccf_path)
                        for plot_name, plot in ind_ccf_plots.items():
                            plot.savefig(f'{ind_ccf_path}/{plot_name}.png')

                    if plot_ind_acfs:
                        ind_acfs_plots = processor.plot_ind_acfs()
                        ind_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                        if not os.path.exists(ind_acf_path):
                            os.makedirs(ind_acf_path)
                        for plot_name, plot in ind_acfs_plots.items():
                            plot.savefig(f'{ind_acf_path}/{plot_name}.png')

                if plot_mean_CCFs:
                    summ_ccf_plots = processor.plot_mean_CCF()
                    for plot_name, plot in summ_ccf_plots.items():
                        plot.savefig(f'{im_save_path}/{plot_name}.png',  dpi=1000)

                if plot_mean_peaks:
                    summ_peak_plots = processor.plot_mean_peak_props()
                    for plot_name, plot in summ_peak_plots.items():
                        plot.savefig(f'{im_save_path}/{plot_name}.png')

                if plot_mean_acfs:
                    summ_acf_plots = processor.plot_mean_ACF()
                    for plot_name, plot in summ_acf_plots.items():
                        plot.savefig(f'{im_save_path}/{plot_name}.png')

                if save_ind_CCF_values:
                    if processor.num_channels == 1:
                        log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                    ind_ccf_val_path = os.path.join(im_save_path, 'Individual_CCF_values')
                    if not os.path.exists(ind_ccf_val_path):
                        os.makedirs(ind_ccf_val_path)
                    processor.save_ind_ccf_values(save_folder=ind_ccf_val_path)

                # Summarize the data for current image as dataframe, and save as .csv
                im_measurements_df = processor.organize_measurements()
                im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index=False)

                # generate summary data for current image
                im_summary_dict = processor.summarize_image(file_name = file_name, group_name = group_name)

                ############################################################
                #stuff to save the line data for graphing on other software 

                # Reshape the indv_line_values array to match the desired CSV format
                reshaped_values = processor.indv_line_values.reshape((processor.num_channels * processor.num_cols, processor.num_rows))

                normalized_values = []
                for line in reshaped_values:
                    normalized_line = (line - np.min(line)) / (np.max(line) - np.min(line))
                    normalized_values.append(normalized_line)

                file_path = im_save_path + '/indv_values.csv'

                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(normalized_values)
                
                ############################################################

                # populate column headers list with keys from the measurements dictionary
                for key in im_summary_dict.keys():
                    if key not in col_headers:
                        col_headers.append(key)

                # append summary data to the summary list
                summary_list.append(im_summary_dict)

                # useless progress bar to force completion of previous bars
                with tqdm(total=10, miniters=1) as dummy_pbar:
                    dummy_pbar.set_description('cleanup:')
                    for i in range(10):
                        dummy_pbar.update(1)

                pbar.update(1)
        
            # create dataframe from summary list
            summary_df = pd.DataFrame(summary_list, columns=col_headers)

            # Get column names for all channels
            channel_cols = [col for col in summary_df.columns if 'Ch' in col and 'Mean Peak Rel Amp' in col]
            # save the summary csv file
            summary_df = summary_df.sort_values('File Name', ascending=True)

            # Normalize mean peak relative amplitude for each channel
            for col in channel_cols:
                channel_mean_rel_amp = summary_df[col]
                norm_mean_rel_amp = channel_mean_rel_amp / channel_mean_rel_amp.min()
                norm_col_name = col.replace('Mean Peak Rel Amp', 'Norm Mean Rel Amp')
                summary_df[norm_col_name] = norm_mean_rel_amp

            summary_df.to_csv(f"{main_save_path}/!{now.strftime('%Y%m%d%H%M')}_summary.csv", index = False)

            # if group names were entered into the gui, generate comparisons between each group
            if group_names != ['']:
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
                        fig = plotComparisons(summary_df, param)
                        fig.savefig(f'{group_save_path}/{param}.png')
                        plt.close(fig)
                    except ValueError:
                        log_params['Plotting errors'].append(f'No data to compare for {param}')

                # save the means for the attributes to make them easier to work with in prism
                processor.save_means_to_csv(main_save_path, group_names, summary_df)

            end = timeit.default_timer()
            log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
            # log parameters and errors
            make_log(main_save_path, log_params)


if __name__ == '__main__':
    main()