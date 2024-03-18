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

from waveanalysis.image_props.image_bin_calc import create_multi_frame_bin_array
from waveanalysis.image_props.image_to_np_arrays import tiff_to_np_array_multi_frame
from waveanalysis.image_props.image_properties import get_multi_frame_properties
from waveanalysis.summarize_save.summarize_rolling import summarize_rolling_file, organize_submovie_measurements

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

    all_images = tiff_to_np_array_multi_frame(folder_path=folder_path)

    print('Processing files...')

    with tqdm(total = len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')

            # Get image properties
            image_path = f'{folder_path}/{file_name}'
            # TODO: add the ability to save the values in terms of seconds if frame_interval is provided
            num_channels, num_frames, frame_interval, pixel_size, pixel_unit = get_multi_frame_properties(image_path=image_path)
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
            bin_values, num_bins, num_x_bins, num_y_bins = create_multi_frame_bin_array(
                                                                kernel_size = box_size, 
                                                                step = box_shift, 
                                                                num_channels = num_channels, 
                                                                num_frames = num_frames, 
                                                                image = all_images[file_name]
                                                            )

            # name without the extension
            name_wo_ext = file_name.rsplit(".",1)[0]

            # Calculate the individual periods for each channel
            indv_periods = np.zeros(shape=(num_submovies, num_channels, num_bins))
            its = num_submovies*num_channels*num_x_bins*num_y_bins
            with tqdm(total = its, miniters=its/100) as pbar:
                pbar.set_description( 'Periods: ')
                for submovie in range(num_submovies):
                    for channel in range(num_channels):
                        for bin in range(num_bins):
                            pbar.update(1)
                            signal = bin_values[roll_by * submovie: roll_size + roll_by * submovie, channel, bin]
                            acf_curve = sp.calc_indv_ACF(signal=signal, num_frames=roll_size, peak_thresh=acf_peak_thresh)
                            period = sp.calc_indv_period(acf_curve=acf_curve, peak_thresh=acf_peak_thresh)

                            indv_periods[submovie, channel, bin] = period
                
            # Calculate the individual peak properties for each channel
            indv_peak_widths = np.zeros(shape=(num_submovies, num_channels, num_bins))
            indv_peak_maxs = np.zeros(shape=(num_submovies, num_channels, num_bins))
            indv_peak_mins = np.zeros(shape=(num_submovies, num_channels, num_bins))
            indv_peak_offsets = np.zeros(shape=(num_submovies, num_channels, num_bins))

            its = num_submovies*num_channels*num_x_bins*num_y_bins
            with tqdm(total = its, miniters=its/100) as pbar:
                pbar.set_description('Peak Props: ')
                for submovie in range(num_submovies):
                    for channel in range(num_channels):
                        for bin in range(num_bins):
                            pbar.update(1)
                            signal = sig.savgol_filter(bin_values[roll_by*submovie : roll_size + roll_by*submovie, channel, bin], window_length=11, polyorder=2)

                            mean_width, mean_max, mean_min, mean_offset, _, _, _, _, _, _, _, _, _ = sp.calc_indv_peak_props(signal=signal)

                            # Store peak measurements for each bin in each channel
                            indv_peak_widths[submovie, channel, bin] = mean_width
                            indv_peak_maxs[submovie, channel, bin] = mean_max
                            indv_peak_mins[submovie, channel, bin] = mean_min
                            indv_peak_amps = indv_peak_maxs - indv_peak_mins
                            indv_peak_rel_amps = indv_peak_amps / indv_peak_mins

            channel_combos = hf.get_channel_combos(num_channels=num_channels)
            num_combos = len(channel_combos)
            # Calculate the individual CCFs and shifts for each channel
            if num_channels > 1:
                indv_shifts = np.zeros(shape=(num_submovies, num_combos, num_bins))
                indv_ccfs = np.zeros(shape=(num_submovies, num_combos, num_bins, roll_size*2-1))
                its = num_submovies*num_combos*num_bins
                with tqdm(total = its, miniters=its/100) as pbar:
                    pbar.set_description( 'Shifts: ')
                    for submovie in range(num_submovies):
                        for combo_number, combo in enumerate(channel_combos):
                            for bin in range(num_bins):
                                pbar.update(1)
                                signal1 = bin_values[roll_by*submovie : roll_size + roll_by*submovie, combo[0], bin]
                                signal2 = bin_values[roll_by*submovie : roll_size + roll_by*submovie, combo[1], bin]
                                ccf = sp.calc_indv_CCF(signal1=signal1, signal2=signal2, num_frames=roll_size)
                                indv_ccfs[submovie, combo_number, bin] = ccf
                                
                                shift = sp.calc_indv_shift(cc_curve=ccf)
                                average_period = np.mean(indv_periods[:, :, bin])
                                shift = sp.small_shifts_correction(delay_frames=shift, average_period=average_period)
                                indv_shifts[submovie, combo_number, bin] = shift

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

            img_parameters_dict = {
                            'Period': indv_periods,
                            'Shift': indv_shifts,
                            'Peak Amp': indv_peak_amps,
                            'Peak Rel Amp': indv_peak_rel_amps,
                            'Peak Width': indv_peak_widths,
                            'Peak Max': indv_peak_maxs,
                            'Peak Min': indv_peak_mins,
                            'Peak Offset': indv_peak_offsets
            }
            
            # TODO: add peak offsets to this function as well
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
            summary_plots = pt.plot_rolling_summary(
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