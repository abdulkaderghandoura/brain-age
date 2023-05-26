def organize_metadata(datasets_path, dataset_name):
    # Split datasets_path into directories
    datasets_path_dirs = datasets_path.split('/' if '/' in datasets_path else '\\')

    if dataset_name == 'hbn':
        raw_data_directory = os.path.join(datasets_path_dirs, 'hbn', 'raw')

        eid_list = []
        for item in os.listdir(raw_data_directory):
            item_path = os.path.join(raw_data_directory, item)
            if os.path.isdir(item_path):
                eid_list.append(item)

        input_metadata_dir = os.path.join(datasets_path_dirs, 'hbn', 'hbn-metadata')

        # List all CSV files in the input_metadata_dir
        csv_files = [file_name for file_name in os.listdir(input_metadata_dir) if file_name.endswith('.csv')]

        # Create an empty DataFrame to store the combined data
        combined_data = pd.DataFrame()

        # Iterate over each CSV file
        for file_name in tqdm(csv_files):
            file_path = os.path.join(input_metadata_dir, file_name)
            data = pd.read_csv(file_path)
            # Append the data to the combined DataFrame
            combined_data = pd.concat([combined_data, data], ignore_index=True)

        # Filter the DataFrame
        combined_data = combined_data[combined_data['EID'].isin(eid_list)]
        # Delete rows with repeated EID values
        combined_data = combined_data[~combined_data['EID'].duplicated()]
        
        combined_data = combined_data.rename(columns={'EID': 'Subject ID'})
        # Select columns of interest
        combined_data = combined_data[['Subject ID','Age', 'Sex']]
        # Map the sex values to meaningful labels ('Males' and 'Females')
        combined_data['Sex'] = combined_data['Sex'].map({0: 'm', 1: 'f'})

        # Save the combined data to a new CSV file
        combined_data.to_csv(metadata_directory0 + 'hbn-metadata.csv', index=False)

        # Save the DataFrame as a .csv file
        output_metadata_path = os.path.join(datasets_path_dirs, 'hbn', 'hbn-metadata.csv')
        combined_data.to_csv(output_metadata_path, index=False)

    elif dataset_name == 'bap':
        input_metadata_path = os.path.join(datasets_path_dirs, 'bap', 'raw', 'clinical_data_updated_2020-08-04.ods')
        
        # Read data from Excel sheets
        file_name1 = 'clinical_data_updated_2020-08-04.ods'
        df1 = pd.read_excel(metadata_directory1 + file_name1, sheet_name='chronic_pain_patients', skiprows=1)
        df2 = pd.read_excel(metadata_directory1 + file_name1, sheet_name='healthy_controls')

        # Concatenate the two dataframes
        df = pd.concat([df1, df2], ignore_index=True)

        # Rename column names
        new_column_names = {'Age (years)': 'Age', 'Sex (m/f)': 'Sex'}
        df = df.rename(columns=new_column_names)

        # Select columns of interest
        df = df[['Subject ID','Age', 'Sex']]

        # Save the DataFrame as a .csv file
        output_metadata_path = os.path.join(datasets_path_dirs, 'bap', 'bap-metadata.csv')
        df.to_csv(output_metadata_path, index=False)

if __name__ == "__main__":
    organize_metadata()
