import os
import timeit
import datetime
import pandas as pd
from tqdm import tqdm
from typing import Any

import waveanalysis.signal_processing as sp
import waveanalysis.housekeeping.housekeeping_functions as hf 

from waveanalysis.image_properties_signal.convert_images import convert_kymos 
from waveanalysis.image_properties_signal.image_properties import get_kymo_image_properties
from waveanalysis.image_properties_signal.create_np_arrays import create_array_from_kymo
from waveanalysis.summarize_organize_savize.add_stats_for_parameter import save_parameter_means_to_csv

from waveanalysis.plotting import (
    plot_indv_peak_props_workflow, 
    plot_indv_acfs_workflow, 
    plot_indv_ccfs_workflow, 
    save_indv_ccfs_workflow, 
    plot_mean_ACFs_workflow, 
    plot_mean_prop_peaks_workflow, 
    plot_mean_CCFs_workflow, 
    save_mean_CCF_values_workflow, 
    generate_group_comparison)
from waveanalysis.summarize_organize_savize.summarize_kymo_standard import (
    organize_standard_kymo_measurements_for_file, 
    summarize_standard_kymo_measurements_for_file)



def kymograph_workflow(
    folder_path: str,
    group_names: list[str],
    log_params: dict[str, Any],
    analysis_type: str,
    box_shift: int,
    subframe_size: int,
    subframe_roll: int,
    line_width: int,
    acf_peak_thresh: float,
    plot_summary_ACFs: bool,
    plot_summary_CCFs: bool,
    plot_summary_peaks: bool,
    plot_indv_ACFs: bool,
    plot_indv_CCFs: bool,
    plot_indv_peaks: bool,
) -> pd.DataFrame:             

    # list of file names in specified directory
    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]
              
    # check for group name errors          
    hf.group_name_error_check(file_names=file_names,
                           group_names=group_names, 
                           log_params=log_params)

    # performance tracker
    start = timeit.default_timer()

    # create main save path
    now = datetime.datetime.now()
    main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
    os.makedirs(main_save_path, exist_ok=True)

    # empty list to fill with summary data for each file
    summary_list = []
    # column headers to use with summary data during conversion to dataframe
    col_headers = []

    # convert images to numpy arrays
    all_images = convert_kymos(folder_path=folder_path)

    print('Processing files...')

    with tqdm(total = len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')

            # TODO: remove the need to set these to None
            num_submovies = None # set to none for now, but will completely remove this parameter in the future
            num_x_bins = None # set to none for now because kymo needs to be none
            num_y_bins = None # set to none for now because kymo needs to be none

            # Get image properties
            image_path = f'{folder_path}/{file_name}'
            num_channels, total_columns, num_frames = get_kymo_image_properties(image_path=image_path, image=all_images[file_name])

            # Create the array for which all future processing will be based on
            bin_values, num_bins = create_array_from_kymo(
                                        line_width = line_width,
                                        total_columns = total_columns,
                                        step = box_shift,
                                        num_channels = num_channels,
                                        num_frames = num_frames,
                                        image = all_images[file_name]
                                    )
            

            # log error and skip image if frames < 2 
            if num_frames < 2:
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
            group_name = hf.match_group_to_file(name_wo_ext=name_wo_ext, group_names=group_names)

            # calculate the individual ACFs for each channel
            indv_acfs, indv_periods = sp.calc_indv_standard_kymo_ACFs_periods(
                num_channels=num_channels, 
                num_bins=num_bins, 
                num_frames=num_frames, 
                bin_values=bin_values, 
                analysis_type=analysis_type,  
                peak_thresh=acf_peak_thresh
                )
                
            # calculate the individual peak properties for each channel
            indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_amps, indv_peak_rel_amps, indv_peak_props = sp.calc_indv_peak_props(
                num_channels=num_channels,
                num_bins=num_bins,
                bin_values=bin_values,
                analysis_type=analysis_type,
                num_submovies=num_submovies,
                roll_by=subframe_roll,
                roll_size=subframe_size,
                num_x_bins=num_x_bins,
                num_y_bins=num_y_bins
            )

            channel_combos = hf.get_channel_combos(num_channels=num_channels)

            # calculate the individual CCFs for each channel
            if num_channels > 1:
                indv_shifts, indv_ccfs = sp.calc_indv_CCFs_shifts_channelCombos(
                    channel_combos=channel_combos,
                    num_bins=num_bins,
                    num_frames=num_frames,
                    bin_values=bin_values,
                    analysis_type=analysis_type,
                    roll_size=subframe_size,
                    roll_by=subframe_roll,
                    num_submovies=num_submovies,
                    periods=indv_periods
                )

            # The code snippet above creates a subfolder within the main save path with the same name as the image file. Will store all associated files in this subfolder
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            os.makedirs(im_save_path, exist_ok=True)

            # plot the mean ACF figures for the file
            if plot_summary_ACFs:
                mean_acf_plots = plot_mean_ACFs_workflow(
                    acfs=indv_acfs,
                    periods=indv_periods,
                    num_frames=num_frames,
                    num_channels=num_channels
                )
                hf.save_plots(mean_acf_plots, im_save_path)

            # plot the mean peak properties figures for the file
            if plot_summary_peaks:
                mean_peak_plots = plot_mean_prop_peaks_workflow(
                    indv_peak_mins=indv_peak_mins,
                    indv_peak_maxs=indv_peak_maxs,
                    indv_peak_amps=indv_peak_amps,
                    indv_peak_widths=indv_peak_widths,
                    num_channels=num_channels
                )
                hf.save_plots(mean_peak_plots, im_save_path)

            # plot the mean CCF figures for the file
            if plot_summary_CCFs and num_channels > 1:
                if num_channels == 1:
                    log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                mean_ccf_plots = plot_mean_CCFs_workflow(
                    signal=indv_ccfs,
                    shifts=indv_shifts,
                    channel_combos=channel_combos,
                    num_frames=num_frames
                )
                hf.save_plots(mean_ccf_plots, im_save_path)

                # save the mean CCF values for the file
                mean_ccf_values = save_mean_CCF_values_workflow(
                    channel_combos=channel_combos,
                    indv_ccfs=indv_ccfs
                )
                hf.save_values_to_csv(mean_ccf_values, im_save_path, indv_ccfs_bool = False)
                # TODO: figure out a way so that the code is not hard coded to the indv vs mean CCFs
            
            # plot the individual ACF figures for the file
            if plot_indv_ACFs:
                indv_acf_plots = plot_indv_acfs_workflow(
                    num_channels=num_channels,
                    num_bins=num_bins,
                    bin_values=bin_values,
                    analysis_type=analysis_type,
                    acfs=indv_acfs,
                    periods=indv_periods,
                    num_frames=num_frames
                )
                indv_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                os.makedirs(indv_acf_path, exist_ok=True)
                hf.save_plots(indv_acf_plots, indv_acf_path)

            # plot the individual peak properties figures for the file
            if plot_indv_peaks:        
                indv_peak_plots = plot_indv_peak_props_workflow(
                    num_channels=num_channels,
                    num_bins=num_bins,
                    bin_values=bin_values,
                    analysis_type=analysis_type,
                    indv_peak_props=indv_peak_props
                )
                indv_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                os.makedirs(indv_peak_path, exist_ok=True)
                hf.save_plots(indv_peak_plots, indv_peak_path)
                
            # plot the individual CCF figures for the file
            if plot_indv_CCFs and num_channels > 1:
                if num_channels == 1:
                    log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'

                indv_ccf_plots = plot_indv_ccfs_workflow(
                    num_bins=num_bins,
                    bin_values=bin_values,
                    analysis_type=analysis_type,
                    channel_combos=channel_combos,
                    indv_shifts=indv_shifts,
                    indv_ccfs=indv_ccfs,
                    num_frames=num_frames
                )
                indv_ccf_plots_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                os.makedirs(indv_ccf_plots_path, exist_ok=True)
                hf.save_plots(indv_ccf_plots, indv_ccf_plots_path)

                # save the individual CCF values for the file
                indv_ccf_values = save_indv_ccfs_workflow(
                    indv_ccfs=indv_ccfs,
                    channel_combos=channel_combos,
                    bin_values=bin_values,
                    analysis_type=analysis_type,
                    num_bins=num_bins
                )
                indv_ccf_val_path = os.path.join(im_save_path, 'Individual_CCF_values')
                os.makedirs(indv_ccf_val_path, exist_ok=True)
                hf.save_values_to_csv(indv_ccf_values, indv_ccf_val_path, indv_ccfs_bool = True)
                # TODO: figure out a way so that the code is not hard coded to the indv vs mean CCFs

                
            # Summarize the data for current image as dataframe, and save as .csv
            im_measurements_df, periods_with_stats, shifts_with_stats, peak_widths_with_stats, peak_maxs_with_stats, peak_mins_with_stats, peak_amps_with_stats, peak_relamp_with_stats = organize_standard_kymo_measurements_for_file(
                num_bins=num_bins,
                num_channels=num_channels,
                channel_combos=channel_combos,
                indv_periods=indv_periods,
                indv_shifts=indv_shifts,
                indv_peak_widths=indv_peak_widths,
                indv_peak_maxs=indv_peak_maxs,
                indv_peak_mins=indv_peak_mins,
                indv_peak_amps=indv_peak_amps,
                indv_peak_rel_amps=indv_peak_rel_amps
            )
            im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)  # type: ignore

            # generate summary data for current image
            im_summary_dict = summarize_standard_kymo_measurements_for_file(
                file_name=file_name, 
                group_name=group_name,
                num_bins=num_bins,
                num_channels=num_channels,
                channel_combos=channel_combos,
                indv_periods=indv_periods,
                periods_with_stats=periods_with_stats,
                indv_shifts=indv_shifts,
                shifts_with_stats=shifts_with_stats,
                indv_peak_widths=indv_peak_widths,
                peak_widths_with_stats=peak_widths_with_stats,
                peak_maxs_with_stats=peak_maxs_with_stats,
                peak_mins_with_stats=peak_mins_with_stats,
                peak_amps_with_stats=peak_amps_with_stats,
                peak_relamp_with_stats=peak_relamp_with_stats
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

        # create dataframe from summary list, then sort and save the summary to a csv file
        summary_df = pd.DataFrame(summary_list, columns=col_headers)
        summary_df = summary_df.sort_values('File Name', ascending=True)
        summary_df.to_csv(f"{main_save_path}/!{now.strftime('%Y%m%d%H%M')}_summary.csv", index = False)

        if group_names != ['']:
            # generate comparisons between each group
            mean_parameter_figs = generate_group_comparison(summary_df = summary_df, 
                                                            log_params = log_params)
            group_plots_save_path = os.path.join(main_save_path, "!group_comparison_graphs")
            os.makedirs(group_plots_save_path, exist_ok=True)
            hf.save_plots(mean_parameter_figs, group_plots_save_path)

            # save the means each parameter for the attributes to make them easier to work with in prism
            parameter_tables_dict = save_parameter_means_to_csv(summary_df=summary_df,
                                                                group_names=group_names)
            mean_measurements_save_path = os.path.join(main_save_path, "!mean_parameter_measurements")
            os.makedirs(mean_measurements_save_path, exist_ok=True)
            for filename, table in parameter_tables_dict.items():
                table.to_csv(f"{mean_measurements_save_path}/{filename}", index = False)

        # performance tracker end
        end = timeit.default_timer()

        # log parameters and errors
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        hf.make_log(main_save_path, log_params)

        return summary_df # only here for testing for now