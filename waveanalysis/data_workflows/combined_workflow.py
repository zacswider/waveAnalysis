import os
import timeit
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
import scipy.signal as sig
import waveanalysis.plotting as pt
import waveanalysis.signal_processing as sp
import waveanalysis.housekeeping.housekeeping_functions as hf 

from waveanalysis.image_props.image_bin_calc import create_multi_frame_bin_array, create_kymo_bin_array
from waveanalysis.image_props.image_to_np_arrays import tiff_to_np_array_multi_frame, tiff_to_np_array_single_frame
from waveanalysis.image_props.image_properties import get_multi_frame_properties, get_single_frame_properties
from waveanalysis.summarize_save.save_stats import save_parameter_means_to_csv, get_mean_CCF_values, get_indv_CCF_values
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
    calc_wave_speeds: bool,
    plot_wave_speeds: bool,
    box_size: int = None,
    box_shift: int = None,
    line_width: int = None,
    frame_interval: float = None,
    pixel_size: float = None
) -> pd.DataFrame:                

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

    # convert images to numpy arrays
    all_images = tiff_to_np_array_multi_frame(folder_path=folder_path) if analysis_type == 'standard' else tiff_to_np_array_single_frame(folder_path=folder_path)

    print('Processing files...')

    with tqdm(total = len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')
            image_path = f'{folder_path}/{file_name}'  
            
            # TODO: add the ability to save the values in terms of seconds if frame_interval is provided

            # Get image properties
            if analysis_type == 'standard':
                num_channels, num_frames, frame_interval, pixel_size, pixel_unit = get_multi_frame_properties(image_path=image_path)
            else: 
                num_channels, num_columns, num_frames, frame_interval, pixel_size, pixel_unit = get_single_frame_properties(image_path=image_path, image=all_images[file_name])

            # log image properties
            log_params['Pixel Size'] = f"{file_name}: {pixel_size} {pixel_unit}s"
            log_params['Frame Interval'] = f"{file_name}: {frame_interval} seconds"

            # log error and skip image if frames < 2; otherwise, log image as processed
            if num_frames < 2:
                print(f"****** ERROR ******",
                    f"\n{file_name} has less than 2 frames",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                continue

            # check if frame interval is not 1 or None and log it
            hf.check_frame_interval(frame_interval=frame_interval, log_params=log_params, file_name=file_name)

            # Create the array of bin values for which all the stats will be calculated
            if analysis_type == 'standard':
                bin_values, num_bins, _, _ = create_multi_frame_bin_array(
                                                                    kernel_size = box_size, 
                                                                    step = box_shift, 
                                                                    num_channels = num_channels, 
                                                                    num_frames = num_frames, 
                                                                    image = all_images[file_name]
                                                                )
            else: # analysis_type == 'kymograph'
                bin_values, num_bins = create_kymo_bin_array(
                                        line_width = line_width,
                                        total_columns = num_columns,
                                        step = box_shift,
                                        num_channels = num_channels,
                                        num_frames = num_frames,
                                        image = all_images[file_name]
                                    )
                if calc_wave_speeds:
                    # user defined wave tracks. will open a window to draw the tracks
                    # wave_tracks = sp.define_wave_tracks(file_path=image_path)
                    wave_tracks = [
                        np.array([[40, 1], [7,  30]]), 
                        np.array([[26, 2], [3,  30]]), 
                        np.array([[9, 22], [12, 30]])
                        ]

                    # check if wave tracks were created and if they are within the image
                    hf.check_if_wave_tracks_created(wave_tracks=wave_tracks, log_params=log_params, file_name=file_name)
                    hf.check_wave_track_coords(wave_tracks=wave_tracks, log_params=log_params, file_name=file_name, num_columns=num_columns, num_frames=num_frames)

                    # calculate the wave speeds form the wave tracks
                    wave_speeds = sp.calc_wave_speeds(wave_tracks=wave_tracks, pixel_size=pixel_size, frame_interval=frame_interval)
            
            # log that the file was processed
            log_params['Files Processed'].append(f'{file_name}')

            name_wo_ext = file_name.rsplit(".",1)[0]
            # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
            group_name = hf.match_group_to_file(name_wo_ext=name_wo_ext, group_names=group_names)

            # get the channel combinations
            channel_combos = hf.get_channel_combos(num_channels=num_channels)
            num_combos = len(channel_combos)

            # initialize arrays to store the individual ACFs, periods, peak properties, CCFs and shifts
            indv_acfs = np.zeros(shape=(num_channels, num_bins, num_frames * 2 - 1))
            indv_periods = np.zeros(shape=(num_channels, num_bins))
            indv_peak_widths = np.zeros(shape=(num_channels, num_bins))
            indv_peak_maxs = np.zeros(shape=(num_channels, num_bins))
            indv_peak_mins = np.zeros(shape=(num_channels, num_bins))
            indv_peak_offsets = np.zeros(shape=(num_channels, num_bins))
            indv_peak_props = {}
            indv_shifts = np.zeros(shape=(num_combos, num_bins))
            indv_ccfs = np.zeros(shape=(num_combos, num_bins, num_frames*2-1))

            # iterate over each channel and bin to calculate the ACFs, periods, peak properties, and CCFs
            for combo_number, combo in enumerate(channel_combos):
                for channel in range(num_channels):
                    for bin in range(num_bins):
                        # calculate the individual ACFs for each channel
                        signal = bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
                        acf_curve = sp.calc_indv_ACF(signal=signal, num_frames=num_frames, peak_thresh=acf_peak_thresh)
                        indv_acfs[channel, bin] = acf_curve

                        # calculate the individual periods for each channel
                        period = sp.calc_indv_period(acf_curve=acf_curve, peak_thresh=acf_peak_thresh)
                        indv_periods[channel, bin] = period

                        # calculate the individual peak properties for each channel
                        smoothed_signal = sig.savgol_filter(signal, window_length = 11, polyorder = 2)                 
                        mean_width, mean_max, mean_min, mean_offset, peaks, proms, heights, leftIndex, rightIndex, midpoints, peak_offsets, left_base, right_base = sp.calc_indv_peak_props(signal=smoothed_signal)
                        indv_peak_widths[channel, bin] = mean_width
                        indv_peak_maxs[channel, bin] = mean_max
                        indv_peak_mins[channel, bin] = mean_min
                        indv_peak_offsets[channel, bin] = mean_offset
                        indv_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': smoothed_signal, 
                                                                'peaks': peaks,
                                                                'proms': proms, 
                                                                'heights': heights, 
                                                                'leftIndex': leftIndex, 
                                                                'rightIndex': rightIndex,
                                                                'midpoints': midpoints,
                                                                'peak_offsets': peak_offsets,
                                                                'left_base': left_base,
                                                                'right_base': right_base}
                        
                        # TODO: rename the keys to be more descriptive

                        # Calculate the individual CCFs and shifts for each channel
                        if num_channels > 1:
                            if analysis_type == 'standard':
                                signal1 = sig.savgol_filter(bin_values[:, combo[0], bin], window_length=11, polyorder=3)
                                signal2 = sig.savgol_filter(bin_values[:, combo[1], bin], window_length=11, polyorder=3)
                            else:
                                signal1 = sig.savgol_filter(bin_values[combo[0], bin], window_length=11, polyorder=3)
                                signal2 = sig.savgol_filter(bin_values[combo[1], bin], window_length=11, polyorder=3)
                            
                            # calculate the individual CCFs for each channel combination
                            ccf = sp.calc_indv_CCF(signal1=signal1, signal2=signal2, num_frames=num_frames)
                            indv_ccfs[combo_number, bin] = ccf

                            # calculate the individual shifts for each channel combination
                            shift = sp.calc_indv_shift(cc_curve=ccf)
                            average_period = np.mean(indv_periods[:, bin]) # If the shift is too small, correct it
                            shift = sp.small_shifts_correction(delay_frames=shift, average_period=average_period)
                            indv_shifts[combo_number, bin] = shift

            # Calculate the peak amplitudes and relative amplitudes
            indv_peak_amps = indv_peak_maxs - indv_peak_mins
            indv_peak_rel_amps = indv_peak_amps / indv_peak_mins

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
            if num_channels > 1:
                img_parameters_dict['Shift'] = indv_shifts

            # add wave speeds to the dictionary if they were calculated
            if calc_wave_speeds:
                img_parameters_dict['Wave Speed'] = wave_speeds    

            # create the directory to save the figures and data for the image
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            os.makedirs(im_save_path, exist_ok=True)

            # plot the mean ACF figures for the file
            if plot_summary_ACFs:
                mean_acf_figs = {}
                for channel in range(num_channels):
                    mean_acf_figs[f'Ch{channel + 1} Mean ACF'] = pt.return_mean_ACF_figure(
                        signal=indv_acfs[channel], 
                        periods=indv_periods[channel], 
                        channel=f'Ch{channel + 1}',
                        num_frames= num_frames)     
                hf.save_plots(mean_acf_figs, im_save_path)

            # plot the mean peak properties figures for the file
            if plot_summary_peaks:
                mean_peak_figs = {}
                for channel in range(num_channels):
                    mean_peak_figs[f'Ch{channel + 1} Peak Props'] = pt.return_mean_prop_peaks_figure(
                        min_array=indv_peak_mins[channel], 
                        max_array=indv_peak_maxs[channel], 
                        amp_array=indv_peak_amps[channel], 
                        width_array=indv_peak_widths[channel], 
                        offsets_array=indv_peak_offsets[channel],
                        Ch_name=f'Ch{channel + 1}')
                hf.save_plots(mean_peak_figs, im_save_path)

            # plot the mean CCF figures for the file
            if plot_summary_CCFs and num_channels > 1:
                mean_ccf_figs = {}
                for combo_number, combo in enumerate(channel_combos):
                    mean_ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = pt.return_mean_CCF_figure(
                        signal=indv_ccfs[combo_number], 
                        shifts=indv_shifts[combo_number], 
                        channel_combo=f'Ch{combo[0] + 1}-Ch{combo[1] + 1}',
                        num_frames= num_frames)
                hf.save_plots(mean_ccf_figs, im_save_path)

                # save the mean CCF values for the file
                mean_ccf_values = get_mean_CCF_values(channel_combos=channel_combos,indv_ccfs=indv_ccfs)
                hf.save_ccf_values_to_csv(mean_ccf_values, im_save_path)

            # Error check for plotting individual CCFs
            elif plot_summary_CCFs and num_channels == 1:
                log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'

            # plot the wave speeds for the file
            if plot_wave_speeds:
                wave_speed_figs = {}
                wave_speed_figs[f'{name_wo_ext} wave speeds'] = pt.return_mean_wave_speeds_figure(wave_speeds=wave_speeds)
                hf.save_plots(wave_speed_figs, im_save_path)
            
            # plot the individual ACF figures for the file
            if plot_indv_ACFs:
                indv_acf_plots = {}
                its = num_channels*num_bins
                with tqdm(total=its, miniters=its/100) as pbar:
                    pbar.set_description('ind acfs')
                    for channel in range(num_channels):
                        for bin in range(num_bins):
                            pbar.update(1) 
                            to_plot = bin_values[:,channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
                            indv_acf_plots[f'Ch{channel + 1} Bin {bin + 1} ACF'] = pt.return_indv_acf_figure(
                                raw_signal=to_plot, 
                                acf_curve=indv_acfs[channel, bin], 
                                Ch_name=f'Ch{channel + 1}', 
                                period=indv_periods[channel, bin],
                                num_frames= num_frames
                                )
                indv_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                os.makedirs(indv_acf_path, exist_ok=True)
                hf.save_plots(indv_acf_plots, indv_acf_path)

            # plot the individual peak properties figures for the file
            if plot_indv_peaks:        
                indv_peak_figs = {}
                its = num_channels*num_bins
                with tqdm(total=its, miniters=its/100) as pbar:
                    pbar.set_description('ind peaks')
                    for channel in range(num_channels):
                        for bin in range(num_bins):
                            pbar.update(1)
                            to_plot = bin_values[:,channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
                            # Generate and store the figure for the current channel and bin
                            indv_peak_figs[f'Ch{channel + 1} Bin {bin + 1} Peak Props'] = pt.return_indv_peak_prop_figure(
                                bin_signal=to_plot,
                                prop_dict=indv_peak_props[f'Ch {channel} Bin {bin}'],
                                Ch_name=f'Ch{channel + 1} Bin {bin + 1}'
                                )
                indv_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                os.makedirs(indv_peak_path, exist_ok=True)
                hf.save_plots(indv_peak_figs, indv_peak_path)
                
            # plot the individual CCF figures for the file
            if plot_indv_CCFs and num_channels > 1:
                if num_channels == 1:
                    log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                indv_ccf_plots = {}
                its = len(channel_combos)*num_bins
                with tqdm(total=its, miniters=its/100) as pbar:
                    pbar.set_description('ind ccfs')
                    for combo_number, combo in enumerate(channel_combos):
                        for bin in range(num_bins):
                            pbar.update(1)
                            if analysis_type == 'standard':
                                to_plot1 = bin_values[:, combo[0], bin] 
                                to_plot2 = bin_values[:, combo[1], bin] 
                            else:
                                to_plot1 = bin_values[combo[0], bin]
                                to_plot2 = bin_values[combo[1], bin]
                            indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = pt.return_indv_ccf_figure(
                                ch1 = pt.normalize_signal(to_plot1),
                                ch2 = pt.normalize_signal(to_plot2),
                                ccf_curve = indv_ccfs[combo_number, bin],
                                ch1_name = f'Ch{combo[0] + 1}',
                                ch2_name = f'Ch{combo[1] + 1}',
                                shift = indv_shifts[combo_number, bin],
                                num_frames = num_frames)
                
                indv_ccf_plots_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                os.makedirs(indv_ccf_plots_path, exist_ok=True)
                hf.save_plots(indv_ccf_plots, indv_ccf_plots_path)

                # save the individual CCF values for the file
                indv_ccf_values = get_indv_CCF_values(
                    indv_ccfs=indv_ccfs,
                    channel_combos=channel_combos,
                    bin_values=bin_values,
                    analysis_type=analysis_type,
                    num_bins=num_bins
                )
                indv_ccf_val_path = os.path.join(im_save_path, 'Individual_CCF_values')
                os.makedirs(indv_ccf_val_path, exist_ok=True)
                hf.save_ccf_values_to_csv(indv_ccf_values, indv_ccf_val_path)                    

            # Summarize the data for current image as dataframe, and save as .csv
            im_measurements_df, parameters_with_stats_dict = summarize_image_standard_kymo(
                num_bins=num_bins,
                num_channels=num_channels,
                channel_combos=channel_combos,
                img_parameters=img_parameters_dict
            )
            im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)  # type: ignore
            
            # generate stats for the image such as mean, median, std, etc
            im_summary_dict = combine_stats_for_image_kymo_standard(
                file_name=file_name, 
                group_name=group_name,
                num_bins=num_bins,
                num_channels=num_channels,
                channel_combos=channel_combos,
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

        return summary_df, wave_tracks # only here for testing for now