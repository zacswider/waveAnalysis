import os
import timeit
import datetime
import pandas as pd
from tqdm import tqdm
from typing import Any
import waveanalysis.plotting as pt
import waveanalysis.signal_processing as sp
import waveanalysis.housekeeping.housekeeping_functions as hf 

from waveanalysis.image_props.image_bin_calc import create_multi_frame_bin_array, create_kymo_bin_array
from waveanalysis.image_props.image_to_np_arrays import tiff_to_np_array_multi_frame, tiff_to_np_array_single_frame
from waveanalysis.image_props.image_properties import get_multi_frame_properties, get_single_frame_properties
from waveanalysis.summarize_save.save_stats import save_parameter_means_to_csv, get_mean_CCF_values, get_indv_CCF_values, save_ccf_values_to_csv
from waveanalysis.summarize_save.summarize_kymo_standard import summarize_image_standard_kymo, combine_stats_for_image_kymo_standard

def combined_workflow(
    folder_path: str,
    group_names: list[str],
    log_params: dict[str, Any],
    analysis_type: str,
    acf_peak_thresh: float,
    plot_summary_ACFs: bool,
    plot_summary_CCFs: bool,
    plot_summary_peaks: bool,
    plot_indv_ACFs: bool,
    plot_indv_CCFs: bool,
    plot_indv_peaks: bool,
    calc_wave_speeds: bool = False,
    plot_wave_speeds: bool = False,
    box_size: int = None,
    bin_shift: int = None, 
    line_width: int = None
) -> pd.DataFrame:
    '''
    This is the combined workflow for kymographs and standard analysis. It processes the image files in the 
    specified folder and saves the summary data and figures to a new folder in the same directory as the 
    image files.

    It functions generally in this order (with some analysis specific steps):
        1. Convert a folder of tiff images to numpy arrays
        2. Iterate over every images in the folder
            a. Get the image properties
            b. Calculate the bin values based on the user provided box/line size and bin shift
            c. Calculate the ACF, period, peak properties, and CCFs/shifts (if specified)
                i. For kymographs, the user will be prompted to define the wave tracks (if specified)
            d. Plot the mean ACF, peak properties, wave speed, and CCF figures (if specified)
            e. Plot the individual ACF, peak properties, and CCF figures (if specified)
            f. Save the summary data and figures to a new folder in the same directory as the image files
        3. Generate the summary data for the entire folder and save it to a csv file
        4. Generate the group comparison figures and save them to a new folder in the same directory as the image files (if group names are specified)
        5. Generate the mean parameter measurements for each group and save them to a new folder in the same directory as the image files (if group names are specified)
        6. Log the parameters and errors to a log file in the new folder

    Parameters:
    - folder_path (str): The path to the folder containing the image files.
    - group_names (list[str]): The list of group names to match with the image files.
    - log_params (dict[str, Any]): The dictionary to store the log parameters.
    - analysis_type (str): The type of analysis to perform ('standard' or 'kymograph').
    - acf_peak_thresh (float): The threshold for detecting peaks in the ACF curve.
    - plot_summary_ACFs (bool): Whether to plot the mean ACF figures for the file.
    - plot_summary_CCFs (bool): Whether to plot the mean CCF figures for the file.
    - plot_summary_peaks (bool): Whether to plot the mean peak properties figures for the file.
    - plot_indv_ACFs (bool): Whether to plot the individual ACF figures for each file.
    - plot_indv_CCFs (bool): Whether to plot the individual CCF figures for each file.
    - plot_indv_peaks (bool): Whether to plot the individual peak properties figures for each file.
    - calc_wave_speeds (bool, optional): Whether to calculate wave speeds. Defaults to False.
    - plot_wave_speeds (bool, optional): Whether to plot the wave speeds. Defaults to False.
    - box_size (int, optional): The size of the box for standard analysis. Defaults to None.
    - bin_shift (int, optional): The shift value for binning. Defaults to None.
    - line_width (int, optional): The width of the line for kymograph analysis. Defaults to None.

    Returns:
    - pd.DataFrame: The summary data for each file.
    '''
    # list of file names in specified directory
    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

    # check for group name errors          
    hf.group_name_error_check(file_names=file_names, group_names=group_names, log_params=log_params)

    # performance tracker
    start = timeit.default_timer()

    # create main save path
    now = datetime.datetime.now()
    main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
    os.makedirs(main_save_path, exist_ok=True)

    # empty list to fill with summary data for each file, and column headers list
    summary_list, col_headers = [], []

    print('Processing files...')

    with tqdm(total = len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')

            ############################################
            ####### Image Convert and Properties #######
            ############################################

            image_path = f'{folder_path}/{file_name}'  
            
            # Get image properties
            if analysis_type == 'standard':
                img_props_dict = get_multi_frame_properties(image_path=image_path)
            else: 
                img_props_dict = get_single_frame_properties(image_path=image_path)

            # check if frame interval is not 1 or None and log it
            frame_interval = hf.check_frame_interval(frame_interval=img_props_dict['frame_interval'], log_params=log_params, file_name=file_name)
            img_props_dict['frame_interval'] = frame_interval

            # add other image properties to the dictionary for later use
            img_props_dict['step'] = bin_shift
            img_props_dict['box_size'] = box_size if analysis_type == 'standard' else None
            img_props_dict['line_width'] = line_width if analysis_type == 'kymograph' else None
            img_props_dict['analysis_type'] = analysis_type
            img_props_dict['peak_thresh'] = acf_peak_thresh

            # log image properties
            log_params['Pixel Size'].append(f"{file_name}: {img_props_dict['pixel_size']} {img_props_dict['pixel_unit']}s")
            log_params['Frame Interval'].append(f"{file_name}: {img_props_dict['frame_interval']} seconds")

            # log error and skip image if frames < 2; otherwise, log image as processed
            if img_props_dict['num_frames'] < 2:
                print(f"****** ERROR ******",
                    f"\n{file_name} has less than 2 frames",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                continue
            # log that the file was processed
            log_params['Files Processed'].append(f'{file_name}')

            # Create the array of bin values for which all the stats will be calculated
            if analysis_type == 'standard':
                image_array = tiff_to_np_array_multi_frame(image_path)
                bin_values, num_bins, _, _ = create_multi_frame_bin_array(image = image_array, 
                                                                          img_props = img_props_dict)
                                
            else: # analysis_type == 'kymograph'
                image_array = tiff_to_np_array_single_frame(image_path)
                bin_values, num_bins = create_kymo_bin_array(image = image_array,
                                                             img_props = img_props_dict)
                
                if calc_wave_speeds:
                    # Have the user define the wave tracks for each kymograph
                    wave_tracks = sp.define_wave_tracks(file_path=image_path)
                    # wave_tracks = [np.array([[40, 1], [7,  30]]), np.array([[26, 2], [3,  30]]), np.array([[9, 22], [12, 30]])] # for testing

                    # check if wave tracks were created and if they are within the image
                    hf.check_if_wave_tracks_created(wave_tracks=wave_tracks, 
                                                    log_params=log_params, 
                                                    file_name=file_name)
                    hf.check_wave_track_coords(wave_tracks=wave_tracks, 
                                               log_params=log_params, 
                                               file_name=file_name, 
                                               num_columns=img_props_dict['num_columns'], 
                                               num_frames=img_props_dict['num_frames'])

                    # calculate the wave speeds form the wave tracks
                    wave_speeds = sp.calc_wave_speeds(wave_tracks=wave_tracks, 
                                                      pixel_size=img_props_dict['pixel_size'], 
                                                      frame_interval=img_props_dict['frame_interval'])
                    
            # get the channel combinations
            channel_combos = hf.get_channel_combos(num_channels=img_props_dict['num_channels'])
            num_combos = len(channel_combos)
            img_props_dict['channel_combos'] = channel_combos
            img_props_dict['num_combos'] = num_combos

            # store the number of bins and the bin values in the image properties dictionary
            img_props_dict['num_bins'] = num_bins
            img_props_dict['bin_values'] = bin_values

            # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
            name_wo_ext = file_name.rsplit(".",1)[0]
            group_name = hf.match_group_to_file(name_wo_ext=name_wo_ext, group_names=group_names)

            ############################################
            ############## Signal Processing ###########
            ############################################

            # Calculate the ACF
            indv_acfs = sp.calc_indv_ACF_workflow(bin_values=bin_values, img_props=img_props_dict)

            # Calculate the period
            indv_periods = sp.calc_indv_period_workflow(acf_curve=indv_acfs, img_props=img_props_dict)

            # Calculate the peak properties
            indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_offsets, indv_peak_props = sp.calc_indv_peak_props_workflow(bin_values=bin_values, img_props=img_props_dict)
            indv_peak_amps = indv_peak_maxs - indv_peak_mins
            indv_peak_rel_amps = indv_peak_amps / indv_peak_mins
            
            # Calculate the individual CCFs and shifts
            if img_props_dict['num_channels'] > 1:
                indv_ccfs = sp.calc_indv_CCF_workflow(bin_values=bin_values, img_props=img_props_dict)
                indv_shifts = sp.calc_indv_shift_workflow(indv_ccfs=indv_ccfs, indv_periods=indv_periods, img_props=img_props_dict)

            # adjust the different waves properties to be the use the frame interval rather than the number of frames
            indv_periods = indv_periods * img_props_dict['frame_interval']
            indv_peak_offsets = indv_peak_offsets * img_props_dict['frame_interval']
            indv_peak_widths = indv_peak_widths * img_props_dict['frame_interval']

            # create dictionary of image parameters and their values for later use
            img_parameters_dict = {
                            'Period': indv_periods,
                            'Peak Amp': indv_peak_amps,
                            'Peak Rel Amp': indv_peak_rel_amps,
                            'Peak Width': indv_peak_widths,
                            'Peak Max': indv_peak_maxs,
                            'Peak Min': indv_peak_mins,
                            'Peak Offset': indv_peak_offsets,
                            }    
            
            # add shifts to the dictionary if there are multiple channels
            if img_props_dict['num_channels'] > 1:
                indv_shifts = indv_shifts * img_props_dict['frame_interval']
                img_parameters_dict['Shift'] = indv_shifts
            # add wave speeds to the dictionary if they were calculated
            if calc_wave_speeds:
                img_parameters_dict['Wave Speed'] = wave_speeds    

            # create the directory to save the figures and data for the image
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            os.makedirs(im_save_path, exist_ok=True)

            ############################################
            ############## Plotting ####################
            ############################################

            # plot the mean ACF figures for the file
            if plot_summary_ACFs:
                mean_acf_figs = pt.plot_mean_ACF_workflow(
                    img_parameters_dict=img_parameters_dict,
                    img_props=img_props_dict,
                    indv_acfs=indv_acfs
                )
                hf.save_plots(mean_acf_figs, im_save_path)

            # plot the mean peak properties figures for the file
            if plot_summary_peaks:
                mean_peak_figs = pt.plot_mean_peak_props_workflow(
                    img_parameters_dict=img_parameters_dict,
                    img_props=img_props_dict
                )
                hf.save_plots(mean_peak_figs, im_save_path)

            # plot the mean CCF figures for the file
            if plot_summary_CCFs and img_props_dict['num_channels'] > 1:
                mean_ccf_figs = pt.plot_mean_CCF_workflow(
                    img_parameters_dict=img_parameters_dict,
                    img_props=img_props_dict,
                    indv_ccfs=indv_ccfs
                )
                hf.save_plots(mean_ccf_figs, im_save_path)
                # save the mean CCF values for the file
                mean_ccf_values = get_mean_CCF_values(channel_combos=channel_combos, indv_ccfs=indv_ccfs, frame_interval=img_props_dict['frame_interval'])
                save_ccf_values_to_csv(mean_ccf_values, im_save_path)

            # Error check for plotting individual CCFs
            elif plot_summary_CCFs and img_props_dict['num_channels'] == 1:
                log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'

            # plot the wave speeds for the file
            if calc_wave_speeds and plot_wave_speeds:
                wave_speed_figs = {}
                wave_speed_figs[f'{name_wo_ext} wave speeds'] = pt.return_mean_wave_speeds_figure(wave_speeds=wave_speeds)
                hf.save_plots(wave_speed_figs, im_save_path)
            
            # plot the individual ACF figures for the file
            if plot_indv_ACFs:
                indv_acf_plots = pt.plot_indv_acf_workflow(
                    bin_values=bin_values,
                    indv_acfs=indv_acfs,
                    img_parameters_dict=img_parameters_dict,
                    img_props=img_props_dict
                )
                indv_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                os.makedirs(indv_acf_path, exist_ok=True)
                hf.save_plots(indv_acf_plots, indv_acf_path)

            # plot the individual peak properties figures for the file
            if plot_indv_peaks:        
                indv_peak_figs = pt.plot_indv_peak_workflow(
                    bin_values=bin_values,
                    img_prop_dict=img_props_dict,
                    indv_peak_props=indv_peak_props,
                    num_frames=img_props_dict['num_frames']
                )
                indv_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                os.makedirs(indv_peak_path, exist_ok=True)
                hf.save_plots(indv_peak_figs, indv_peak_path)
                
            # plot the individual CCF figures for the file
            if plot_indv_CCFs and img_props_dict['num_channels'] > 1:
                if img_props_dict['num_channels'] == 1:
                    log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                indv_ccf_plots = pt.plot_indv_ccf_workflow(
                    bin_values=bin_values,
                    indv_ccfs=indv_ccfs,
                    img_parameters_dict=img_parameters_dict,
                    img_props=img_props_dict
                )
                indv_ccf_plots_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                os.makedirs(indv_ccf_plots_path, exist_ok=True)
                hf.save_plots(indv_ccf_plots, indv_ccf_plots_path)
                # save the individual CCF values for the file
                indv_ccf_values = get_indv_CCF_values(
                    indv_ccfs=indv_ccfs,
                    bin_values=bin_values,
                    img_props_dict=img_props_dict
                )
                indv_ccf_val_path = os.path.join(im_save_path, 'Individual_CCF_values')
                os.makedirs(indv_ccf_val_path, exist_ok=True)
                save_ccf_values_to_csv(indv_ccf_values, indv_ccf_val_path)                    

            ############################################
            ############## Saving ######################
            ############################################

            # Summarize the data for current image as dataframe, and save as .csv
            im_measurements_df, parameters_with_stats_dict = summarize_image_standard_kymo(
                img_parameters=img_parameters_dict,
                img_props_dict=img_props_dict
            )
            im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)  # type: ignore
            
            # generate stats for the image such as mean, median, std, etc
            im_summary_dict = combine_stats_for_image_kymo_standard(
                file_name=file_name, 
                group_name=group_name,
                img_props=img_props_dict,
                img_parameters_dict=img_parameters_dict,
                parameters_with_stats_dict=parameters_with_stats_dict
            )

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

        ############################################
        ############## Summary #####################
        ############################################

        # create dataframe from summary list, then sort and save the summary to a csv file
        summary_df = pd.DataFrame(summary_list, columns=col_headers)
        summary_df = summary_df.sort_values('File Name', ascending=True)
        summary_df.to_csv(f"{main_save_path}/!{now.strftime('%Y%m%d%H%M')}_summary.csv", index = False)

        if group_names != ['']:
            # generate comparisons between each group
            mean_parameter_figs = pt.generate_group_comparison(summary_df = summary_df, log_params = log_params)
            group_plots_save_path = os.path.join(main_save_path, "!group_comparison_graphs")
            os.makedirs(group_plots_save_path, exist_ok=True)
            hf.save_plots(mean_parameter_figs, group_plots_save_path)

            # save the means each parameter for the attributes to make them easier to work with 
            parameter_tables_dict = save_parameter_means_to_csv(summary_df=summary_df,group_names=group_names)
            mean_measurements_save_path = os.path.join(main_save_path, "!mean_parameter_measurements")
            os.makedirs(mean_measurements_save_path, exist_ok=True)
            for filename, table in parameter_tables_dict.items():
                table.to_csv(f"{mean_measurements_save_path}/{filename}", index = False)

        # performance tracker end
        end = timeit.default_timer()

        # log parameters and errors
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        hf.make_log(main_save_path, log_params)

        return summary_df # only here for testing