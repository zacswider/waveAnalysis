import os
import timeit
import datetime
import pandas as pd
from tqdm import tqdm
from typing import Any

import waveanalysis.signal_processing as sp
import waveanalysis.housekeeping.housekeeping_functions as hf

from waveanalysis.plotting import plot_rolling_summary
from waveanalysis.image_properties_signal.convert_images import convert_movies  
from waveanalysis.image_properties_signal.image_properties import get_image_properties
from waveanalysis.image_properties_signal.create_np_arrays import create_array_from_standard_rolling
from waveanalysis.summarize_organize_savize.summarize_rolling import summarize_rolling_file, organize_submovie_measurements

def rolling_workflow(
    folder_path: str,
    log_params: dict[str, Any],
    box_size: int,
    box_shift: int,
    roll_size: int,
    roll_by: int,
    acf_peak_thresh: float
) -> pd.DataFrame:             

    # list of file names in specified directory
    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

    # performance tracker
    start = timeit.default_timer()

    # create main save path
    now = datetime.datetime.now()
    main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
    os.makedirs(main_save_path, exist_ok=True)

    # list of file names in specified directory
    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

    all_images = convert_movies(folder_path=folder_path)

    print('Processing files...')

    with tqdm(total = len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')

            # Get image properties
            image_path = f'{folder_path}/{file_name}'
            num_channels, num_frames, frame_interval, pixel_size, pixel_unit = get_image_properties(image_path=image_path)
            assert isinstance(roll_size, int) and isinstance(roll_by, int), 'Roll size and roll by must be integers'
            num_submovies = (num_frames - roll_size) // roll_by

            # log error and skip image if frames < 2 
            if num_frames < 2:
                print(f"****** ERROR ******",
                    f"\n{file_name} has less than 2 frames",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                continue
            log_params['Files Processed'].append(f'{file_name}')
            
            # Create the array for which all future processing will be based on
            bin_values, num_bins, num_x_bins, num_y_bins = create_array_from_standard_rolling(
                                                                kernel_size = box_size, 
                                                                step = box_shift, 
                                                                num_channels = num_channels, 
                                                                num_frames = num_frames, 
                                                                image = all_images[file_name]
                                                            )

            # name without the extension
            name_wo_ext = file_name.rsplit(".",1)[0]

            # calculate the individual ACFs for each channel
            indv_acfs, indv_periods = sp.calc_indv_rolling_ACFs_periods(
                num_channels=num_channels, 
                num_bins=num_bins, 
                bin_values=bin_values, 
                roll_size=roll_size, 
                roll_by=roll_by, 
                num_submovies=num_submovies, 
                num_x_bins=num_x_bins, 
                num_y_bins=num_y_bins, 
                peak_thresh=acf_peak_thresh
                )
                
            # calculate the individual peak properties for each channel
            indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_amps, indv_peak_rel_amps =sp.calc_indv_peak_props_rolling(
                num_channels=num_channels,
                num_bins=num_bins,
                bin_values=bin_values,
                num_submovies=num_submovies,
                roll_by=roll_by,
                roll_size=roll_size,
                num_x_bins=num_x_bins,
                num_y_bins=num_y_bins
            )

            channel_combos = hf.get_channel_combos(num_channels=num_channels)

            # calculate the individual CCFs for each channel
            if num_channels > 1:
                indv_shifts, indv_ccfs = sp.calc_indv_CCFs_shifts_rolling(
                    channel_combos = channel_combos,
                    num_bins=num_bins,
                    bin_values=bin_values,
                    roll_size=roll_size,
                    roll_by=roll_by,
                    num_submovies=num_submovies,
                    periods=indv_periods
                )

            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            os.makedirs(im_save_path, exist_ok=True)

            # calculate the number of subframes used
            log_params['Submovies Used'].append(num_submovies)

            # summarize the data for each subframe as individual dataframes, and save as .csv
            submovie_meas_list = organize_submovie_measurements(
                num_bins=num_bins,
                num_channels=num_channels,
                num_submovies=num_submovies,
                indv_periods=indv_periods,
                indv_ccfs=indv_ccfs,
                indv_peak_widths=indv_peak_widths,
                indv_peak_maxs=indv_peak_maxs,
                indv_peak_mins=indv_peak_mins,
                indv_peak_amps=indv_peak_amps,
                indv_peak_rel_amps=indv_peak_rel_amps,
                channel_combos=channel_combos
            )
            csv_save_path = os.path.join(im_save_path, 'rolling_measurements')
            os.makedirs(csv_save_path, exist_ok=True)
            for measurement_index, submovie_meas_df in enumerate(submovie_meas_list):  # type: ignore
                submovie_meas_df.to_csv(f'{csv_save_path}/{name_wo_ext}_subframe{measurement_index}_measurements.csv', index = False)
            
            # summarize the data for each subframe as a single dataframe, and save as .csv
            summary_df = summarize_rolling_file(
                num_bins=num_bins,
                num_channels=num_channels,
                channel_combos=channel_combos,
                num_submovies=num_submovies,
                indv_periods=indv_periods,
                indv_shifts=indv_shifts,
                indv_peak_widths=indv_peak_widths,
                indv_peak_maxs=indv_peak_maxs,
                indv_peak_mins=indv_peak_mins,
                indv_peak_amps=indv_peak_amps,
                indv_ccfs=indv_ccfs
            )
            summary_df.to_csv(f'{im_save_path}/{name_wo_ext}_summary.csv', index = False)

            # make and save the summary plot for rolling data
            summary_plots = plot_rolling_summary(
                num_channels=num_channels,
                fullmovie_summary=summary_df,
                channel_combos=channel_combos
            )
            plot_save_path = os.path.join(im_save_path, 'summary_plots')
            os.makedirs(plot_save_path, exist_ok=True)
            hf.save_plots(summary_plots, plot_save_path)

            end = timeit.default_timer()
            log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
            # log parameters and errors
            hf.make_log(main_save_path, log_params)

            pbar.update(1)

            if name_wo_ext == '1_Group2':
                return summary_df # only return this now for testing purposes. Will remove later


            
