import os
import csv
import timeit
import datetime
import pandas as pd
from tqdm import tqdm
from typing import Any

import waveanalysis.image_signals as sc
from waveanalysis.waveanalysismods.processor import TotalSignalProcessor
from waveanalysis.housekeeping.housekeeping_functions import make_log

# TODO: major refactor in how the rolling movies are generated and processed so that we can use the same functions for both standard and rolling

def rolling_workflow(
    folder_path: str,
    log_params: dict[str, Any],
    analysis_type: str,
    box_size: int,
    box_shift: int,
    subframe_size: int,
    subframe_roll: int,
    line_width: int,
    acf_peak_thresh: float
) -> pd.DataFrame:               
                
    start = timeit.default_timer()

    all_images = sc.convert_movies(folder_path=folder_path)

    # list of file names in specified directory
    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

    # create main save path
    now = datetime.datetime.now()
    main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    # empty list to fill with summary data for each file
    summary_list = []
    # column headers to use with summary data during conversion to dataframe
    col_headers = []

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
            processor.calc_indv_peak_props()
            processor.calc_indv_ACFs(peak_thresh = acf_peak_thresh)
            if processor.num_channels > 1:
                processor.calc_indv_CCFs()

            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)

            # calculate the number of subframes used
            num_submovies = processor.num_submovies
            log_params['Submovies Used'].append(num_submovies)

            # summarize the data for each subframe as individual dataframes, and save as .csv
            submovie_meas_list = processor.organize_measurements()
            csv_save_path = os.path.join(im_save_path, 'rolling_measurements')
            if not os.path.exists(csv_save_path):
                os.makedirs(csv_save_path)
            for measurement_index, submovie_meas_df in enumerate(submovie_meas_list):  # type: ignore
                submovie_meas_df: pd.DataFrame
                submovie_meas_df.to_csv(f'{csv_save_path}/{name_wo_ext}_subframe{measurement_index}_measurements.csv', index = False)
            
            # summarize the data for each subframe as a single dataframe, and save as .csv
            summary_df = processor.summarize_rolling_file()
            summary_df.to_csv(f'{im_save_path}/{name_wo_ext}_summary.csv', index = False)

            # make and save the summary plot for rolling data
            summary_plots = processor.plot_rolling_summary()
            plot_save_path = os.path.join(im_save_path, 'summary_plots')
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)
            for title, plot in summary_plots.items():
                plot.savefig(f'{plot_save_path}/{name_wo_ext}_{title}.png')

            # generate summary data for current image
            im_summary_dict = processor.summarize_image(file_name = file_name)

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
        summary_df = pd.DataFrame(summary_list, columns=col_headers)

        # save the summary csv file
        summary_df = summary_df.sort_values('File Name', ascending=True)

        summary_df.to_csv(f"{main_save_path}/!{now.strftime('%Y%m%d%H%M')}_summary.csv", index = False)

        end = timeit.default_timer()
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        # log parameters and errors
        make_log(main_save_path, log_params)

        return summary_df


