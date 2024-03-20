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
) -> dict:

    mean_ccf_values = {}

    for combo_number, combo in enumerate(channel_combos):
        arr_mean = np.nanmean(indv_ccfs[combo_number], axis = 0)
        arr_std = np.nanstd(indv_ccfs[combo_number], axis = 0)
        # Combine mean and standard deviation into a list of tuples
        mean_ccf_values[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF values'] = list(zip_longest(range(1, len(arr_mean) + 1), arr_mean, arr_std, fillvalue=None))

    return mean_ccf_values

def get_indv_CCF_values(
    indv_ccfs:np.ndarray,
    channel_combos:np.ndarray,
    bin_values:np.ndarray,
    analysis_type:str,
    num_bins:int
) -> dict:
    
    indv_ccf_values = {}

    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):      
            # Save the individual bin values
            to_plot1 = bin_values[:, combo[0], bin] if analysis_type == "standard" else bin_values[combo[0], bin]
            to_plot2 = bin_values[:, combo[1], bin] if analysis_type == "standard" else bin_values[combo[1], bin]
            ccf_curve = indv_ccfs[combo_number, bin]
            measurements = list(zip_longest(range(1, len(ccf_curve) + 1),  normalize_signal(to_plot1), normalize_signal(to_plot2), ccf_curve, fillvalue=None))

            indv_ccf_values[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = measurements
            
    return indv_ccf_values

def save_ccf_values_to_csv(
    values: dict, 
    path: str,
):
    for filename, measurements in values.items():
        file_path = os.path.join(path, f'{filename}.csv')
        headers, data = determine_structure_and_values(measurements)
        write_to_csv(file_path, headers, data)

def determine_structure_and_values(measurements: Union[List[Tuple], List[List]]) -> Tuple[List[str], List[Tuple]]:
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

def write_to_csv(file_path: str, headers: List[str], data: List[Tuple]):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)