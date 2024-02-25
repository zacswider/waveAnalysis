import os
import csv
import timeit
import datetime
import pandas as pd
from tqdm import tqdm
from typing import Any

from waveanalysis.image_properties_signal.convert_images import convert_movies  
from waveanalysis.waveanalysismods.processor import TotalSignalProcessor
from waveanalysis.housekeeping.housekeeping_functions import make_log, group_name_error_check, check_and_make_save_path, save_plots

def rolling_workflow(
    folder_path: str,
    group_names: list[str],
    log_params: dict[str, Any],
    analysis_type: str,
    box_size: int,
    box_shift: int,
    subframe_size: int,
    subframe_roll: int,
    line_width: int,
    acf_peak_thresh: float
) -> pd.DataFrame:             

    # list of file names in specified directory
    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]
              
    group_name_error_check(file_names=file_names,
                           group_names=group_names, 
                           log_params=log_params)

    # performance tracker
    start = timeit.default_timer()
    # create main save path
    now = datetime.datetime.now()

    main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
    check_and_make_save_path(main_save_path)

    all_images = convert_movies(folder_path=folder_path)

    print('Processing files...')

    with tqdm(total = len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')
            processor = TotalSignalProcessor(analysis_type = analysis_type, 
                                             image_path = f'{folder_path}/{file_name}',
                                             image = all_images[file_name], 
                                             kern = box_size, 
                                             step = box_shift, 
                                             roll_size = subframe_size, 
                                             roll_by = subframe_roll, 
                                             line_width = line_width)
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

            # calculate the population signal properties
            processor.calc_indv_ACFs(peak_thresh = acf_peak_thresh)
            processor.calc_indv_peak_props()
            if processor.num_channels > 1:
                processor.calc_indv_CCFs()

            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            check_and_make_save_path(im_save_path)

            # calculate the number of subframes used
            num_submovies = processor.num_submovies
            log_params['Submovies Used'].append(num_submovies)

            # summarize the data for each subframe as individual dataframes, and save as .csv
            submovie_meas_list = processor.get_submovie_measurements()
            csv_save_path = os.path.join(im_save_path, 'rolling_measurements')
            check_and_make_save_path(csv_save_path)
            for measurement_index, submovie_meas_df in enumerate(submovie_meas_list):  # type: ignore
                submovie_meas_df.to_csv(f'{csv_save_path}/{name_wo_ext}_subframe{measurement_index}_measurements.csv', index = False)
            
            # summarize the data for each subframe as a single dataframe, and save as .csv
            summary_df = processor.summarize_rolling_file()
            summary_df.to_csv(f'{im_save_path}/{name_wo_ext}_summary.csv', index = False)

            # make and save the summary plot for rolling data
            summary_plots = processor.plot_rolling_summary()
            plot_save_path = os.path.join(im_save_path, 'summary_plots')
            check_and_make_save_path(plot_save_path)
            save_plots(summary_plots, plot_save_path)

            end = timeit.default_timer()
            log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
            # log parameters and errors
            make_log(main_save_path, log_params)

            pbar.update(1)

            if name_wo_ext == '1_Group2':
                return summary_df # only return this now for testing purposes. Will remove later


            
