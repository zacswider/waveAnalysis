import pandas as pd

def save_parameter_means_to_csv(
    summary_df: pd.DataFrame,
    group_names: list,
) -> dict:
    parameter_tables_dict = {}

    # extract the parameters to save mean values for
    parameters_to_extract = []
    for column in summary_df.columns:
        if 'Mean' in column:
            parameters_to_extract.append(column)

    # Extract data for each group and parameter, save to a dataframe
    for parameter in parameters_to_extract:
        indv_parameter_table = pd.DataFrame(columns=['Data Type', 'Group Name', 'Value'])
        filename = f"{parameter.lower().replace(' ', '_')}_means.csv"
        for group_name in group_names:
            group_data = summary_df.loc[summary_df['File Name'].str.contains(group_name)]
            values = group_data[parameter].tolist()            
            indv_parameter_table = pd.concat([indv_parameter_table, pd.DataFrame({'Data Type': parameter, 'Group Name': group_name, 'Value': values})], ignore_index=True)

        indv_parameter_table = pd.pivot_table(indv_parameter_table, index=indv_parameter_table.index, columns='Group Name', values='Value')
        for col in indv_parameter_table.columns:
            indv_parameter_table[col] = sorted(indv_parameter_table[col], key=lambda x: 1 if pd.isna(x) or x == '' else 0)
        
        parameter_tables_dict[filename] = indv_parameter_table

    return parameter_tables_dict