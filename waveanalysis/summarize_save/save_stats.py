import os
import csv
import numpy as np
import pandas as pd
from itertools import zip_longest
from typing import Union, List, Tuple
from waveanalysis.signal_processing import normalize_signal

def save_parameter_means_to_csv(
    summary_df: pd.DataFrame,
    group_names: list,
) -> dict:
    '''
    Save the mean values of parameters for each group to separate CSV files.

    Parameters:
        summary_df (pd.DataFrame): The summary dataframe containing the parameter values.
        group_names (list): A list of group names.

    Returns:
        dict: A dictionary containing the filenames as keys and the corresponding parameter tables as values.
    '''
    # create a dictionary to store the parameter tables
    parameter_tables_dict = {}
    parameters_to_extract = [column for column in summary_df.columns if 'Mean' in column]

    # extract the mean values for each group
    for parameter in parameters_to_extract:
        # create a dataframe to store the mean values for each group
        individual_parameter_table = pd.DataFrame(columns=['Data Type', 'Group Name', 'Value'])
        filename = f"{parameter.lower().replace(' ', '_')}_means.csv"

        # extract the mean values for each group
        for group_name in group_names:
            group_data = summary_df.loc[summary_df['File Name'].str.contains(group_name)]
            values = group_data[parameter].tolist()
            df_to_concat = pd.DataFrame({'Data Type': parameter, 'Group Name': group_name, 'Value': values})
            # concatenate the dataframes
            if not individual_parameter_table.empty:
                individual_parameter_table = pd.concat([individual_parameter_table, df_to_concat], ignore_index=True)
            else:
                individual_parameter_table = df_to_concat

        # pivot the table to have the group names as columns
        individual_parameter_table = pd.pivot_table(individual_parameter_table, 
                                                    index=individual_parameter_table.index, 
                                                    columns='Group Name', 
                                                    values='Value')
        
        # Sort the columns by group name and replace NaNs with empty strings
        individual_parameter_table = individual_parameter_table.apply(
            lambda col: sorted(col, key=lambda x: 1 if pd.isna(x) or x == '' else 0)
        )
        
        parameter_tables_dict[filename] = individual_parameter_table

    return parameter_tables_dict

def get_mean_CCF_values(
    channel_combos: list, 
    indv_ccfs: np.ndarray, 
    frame_interval: float
) -> dict:
    '''
    Calculate the mean cross-correlation function (CCF) values for each channel combination.

    Args:
        channel_combos (list): A list of channel combinations.
        indv_ccfs (np.ndarray): An array of individual CCFs.
        frame_interval (float): The time interval between frames.

    Returns:
        dict: A dictionary containing the mean CCF values for each channel combination.
              The keys are in the format 'ChX-ChY Mean CCF values', where X and Y are the channel numbers.
              The values are lists of tuples, where each tuple contains the frame index, mean CCF value, and standard deviation.
    '''
    # Initialize dictionary to store the mean CCF values
    mean_ccf_values = {}

    # Loop through each channel combination
    for combo_number, combo in enumerate(channel_combos):
        # Calculate the mean and standard deviation of the CCF values
        arr_mean = np.nanmean(indv_ccfs[combo_number], axis=0)
        arr_std = np.nanstd(indv_ccfs[combo_number], axis=0)
        # Create a list of frame indices
        arr_list = [i * frame_interval for i in range(len(arr_mean))]
        # Combine mean and standard deviation into a list of tuples
        mean_ccf_values[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF values'] = list(zip_longest(arr_list, arr_mean, arr_std, fillvalue=None))

    return mean_ccf_values

def get_indv_CCF_values(
    indv_ccfs:np.ndarray,
    bin_values:np.ndarray,
    img_props_dict: dict
) -> dict:
    '''
    Calculate and return the individual cross-correlation function (CCF) values for each channel combination and bin.

    Parameters:
    - indv_ccfs (np.ndarray): Array of individual CCF curves for each channel combination and bin.
    - bin_values (np.ndarray): Array of bin values for each channel combination and bin.
    - img_props_dict (dict): Dictionary containing image properties such as frame interval, number of bins, analysis type, and channel combinations.

    Returns:
    - indv_ccf_values (dict): Dictionary containing the individual CCF values for each channel combination and bin.
    '''
    # Extract image properties from the dictionary
    frame_interval = img_props_dict['frame_interval']
    num_bins = img_props_dict['num_bins']
    analysis_type = img_props_dict['analysis_type']
    channel_combos = img_props_dict['channel_combos']
    
    # Initialize dictionary to store the individual CCF values
    indv_ccf_values = {}

    # Loop through each channel combination and bin
    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):      
            # Save the individual bin values
            to_plot1 = bin_values[:, combo[0], bin] if analysis_type == "standard" else bin_values[combo[0], bin]
            to_plot2 = bin_values[:, combo[1], bin] if analysis_type == "standard" else bin_values[combo[1], bin]
            # Create a list of tuples containing the time, channel 1 value, channel 2 value, and CCF value
            ccf_curve = indv_ccfs[combo_number, bin]
            arr_list = [i * frame_interval for i in range(len(ccf_curve))]
            measurements = list(zip_longest(arr_list,  normalize_signal(to_plot1), normalize_signal(to_plot2), ccf_curve, fillvalue=None))

            indv_ccf_values[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = measurements
            
    return indv_ccf_values

def save_ccf_values_to_csv(
    values: dict, 
    path: str,
) -> None:
    '''
    Save the CCF values to CSV files.
    '''
    for filename, measurements in values.items():
        file_path = os.path.join(path, f'{filename}.csv')
        headers, data = determine_structure_and_values(measurements)
        write_to_csv(file_path, headers, data)

def determine_structure_and_values(measurements: Union[List[Tuple], List[List]]) -> Tuple[List[str], List[Tuple]]:
    '''
    Determine the structure of the measurements and return the headers and values.
    '''
    # Check the structure of the measurements to determine headers and values
    first_entry = measurements[0]
    if len(first_entry) == 4:
        # Individual CCFs
        headers = ['Time', 'Ch1_Value', 'Ch2_Value', 'CCF_Value']
    elif len(first_entry) == 3:
        # Mean CCFs
        headers = ['Time', 'Mean', 'StDev']
    else:
        raise ValueError("Unsupported measurements format")

    return headers, measurements

def write_to_csv(file_path: str, headers: List[str], data: List[Tuple]) -> None:
    '''
    Write the headers and data to a CSV file.
    '''
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)