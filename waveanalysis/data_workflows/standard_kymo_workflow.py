import os
import csv
import timeit
import datetime
import pandas as pd
from tqdm import tqdm
from typing import Any

import waveanalysis.image_signals as sc
from waveanalysis.waveanalysismods.processor import TotalSignalProcessor
from waveanalysis.housekeeping.housekeeping_functions import make_log, generate_group_comparison, ensure_group_names

def standard_kymo_workflow(
    folder_path: str,
    group_names: list[str],
    log_params: dict[str, Any],
    analysis_type: str,
    box_size: int,
    box_shift: int,
    subframe_size: int,
    subframe_roll: int,
    line_width: int,
    acf_peak_thresh: float,
    plot_summary_ACFs: bool,
    plot_summary_CCFs: bool,
    plot_summary_peaks: bool,
    plot_ind_ACFs: bool,
    plot_ind_CCFs: bool,
    plot_ind_peaks: bool,
) -> pd.DataFrame:                           
    start = timeit.default_timer()

    # convert images to numpy arrays
    if analysis_type == 'kymograph':
        all_images = sc.convert_kymos(folder_path=folder_path)
    else:
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

    # error checking for group names
    ensure_group_names(folder_path=folder_path, file_names=file_names, group_names=group_names, log_params=log_params)

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

            # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
            group_name = None
            if group_names != ['']:
                try:
                    group_name = [group for group in group_names if group in name_wo_ext][0]
                except IndexError:
                    pass

            # calculate the population signal properties
            processor.calc_indv_ACFs(peak_thresh = acf_peak_thresh)
            processor.calc_indv_peak_props()
            if processor.num_channels > 1:
                processor.calc_indv_CCFs()

            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)

            # if standard or kymograph analysis
            if analysis_type != "rolling":
                # plot and save the mean autocorrelation, crosscorrelation, and peak properties for each channel
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
                
                # plot and save the individual autocorrelation, crosscorrelation, and peak properties for each bin in channel
                if plot_ind_peaks:        
                    ind_peak_plots = processor.plot_indv_peak_props()
                    ind_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                    if not os.path.exists(ind_peak_path):
                        os.makedirs(ind_peak_path)
                    for plot_name, plot in ind_peak_plots.items():
                        plot.savefig(f'{ind_peak_path}/{plot_name}.png')

                if plot_ind_ACFs:
                    ind_acf_plots = processor.plot_indv_acfs()
                    ind_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                    if not os.path.exists(ind_acf_path):
                        os.makedirs(ind_acf_path)
                    for plot_name, plot in ind_acf_plots.items():
                        plot.savefig(f'{ind_acf_path}/{plot_name}.png')

                if plot_ind_CCFs and processor.num_channels > 1:
                    if processor.num_channels == 1:
                        log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                    ind_ccf_val_path = os.path.join(im_save_path, 'Individual_CCF_values')
                    if not os.path.exists(ind_ccf_val_path):
                        os.makedirs(ind_ccf_val_path)
                    ind_ccf_plots = processor.plot_indv_ccfs(save_folder=ind_ccf_val_path)
                    ind_ccf_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                    if not os.path.exists(ind_ccf_path):
                        os.makedirs(ind_ccf_path)
                    for plot_name, plot in ind_ccf_plots.items():
                        plot.savefig(f'{ind_ccf_path}/{plot_name}.png')

                # Summarize the data for current image as dataframe, and save as .csv
                im_measurements_df = processor.organize_measurements()
                im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)  # type: ignore

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
        summary_df = pd.DataFrame(summary_list, columns=col_headers)

        # save the summary csv file
        summary_df = summary_df.sort_values('File Name', ascending=True)

        summary_df.to_csv(f"{main_save_path}/!{now.strftime('%Y%m%d%H%M')}_summary.csv", index = False)

        # if group names were entered into the gui, generate comparisons between each group
        if group_names != ['']:
            generate_group_comparison(main_save_path = main_save_path, processor = processor, summary_df = summary_df, log_params = log_params)
            
            # save the means each parameter for the attributes to make them easier to work with in prism
            processor.save_parameter_means_to_csv(main_save_path, group_names, summary_df)

        end = timeit.default_timer()
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        # log parameters and errors
        make_log(main_save_path, log_params)

        return summary_df


