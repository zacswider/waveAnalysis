import pandas as pd

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
            individual_parameter_table = pd.concat([individual_parameter_table, 
                                        pd.DataFrame({'Data Type': parameter, 
                                                        'Group Name': group_name, 
                                                        'Value': values})], 
                                        ignore_index=True)

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
