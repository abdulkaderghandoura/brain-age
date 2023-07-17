import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def organize_metadata(datasets_path, dataset_name):
    """Organizes metadata for the given dataset.

    Args:
        datasets_path (str or Path): The path to the datasets directory.
        dataset_name (str): The name of the dataset to organize metadata for.
    """
    datasets_path = Path(datasets_path).resolve()
    
    raw_dir_name = {'hbn': 'raw', 'bap': 'raw', 'lemon': 'LEMON_RAW'}
    raw_dir_path = datasets_path / dataset_name / raw_dir_name[dataset_name]

    if dataset_name == 'hbn':
        
        input_metadata_dir = datasets_path / 'hbn' / 'hbn-metadata'
        # List all CSV files in the input_metadata_dir
        csv_files = [file_path.name for file_path in input_metadata_dir.glob('*.csv')]
        
        combined_data = pd.DataFrame()
        for file_name in tqdm(csv_files):
            file_path = input_metadata_dir / file_name
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)

        eid_list = [entry.name for entry in raw_dir_path.iterdir() if entry.is_dir()]

        # Filter the DataFrame
        combined_data = combined_data[combined_data['EID'].isin(eid_list)]
        # Delete rows with repeated EID values
        combined_data = combined_data[~combined_data['EID'].duplicated()]
        
        combined_data = combined_data.rename(columns={'EID': 'Subject ID'})
        # Select columns of interest
        combined_data = combined_data[['Subject ID','Age', 'Sex']]
        # Map the sex values to meaningful labels ('m' for males and 'f' females)
        combined_data['Sex'] = combined_data['Sex'].map({0: 'm', 1: 'f'})

        output_metadata_path = datasets_path / 'hbn' / 'hbn-metadata.csv'
        combined_data.to_csv(output_metadata_path, index=False)

    elif dataset_name == 'bap':        
        # Read data from Excel sheets
        input_metadata_path = list(raw_dir_path.glob('*.ods'))[0]
        df1 = pd.read_excel(input_metadata_path, sheet_name='chronic_pain_patients', skiprows=1)
        df2 = pd.read_excel(input_metadata_path, sheet_name='healthy_controls')

        # Rename column names
        # ** Note: unline the healthy_controls sheet, no space in 'Age(years)' in chronic_pain_patients sheet**
        df1 = df1.rename(columns={'Age(years)': 'Age', 'Sex (m/f)': 'Sex'})
        df2 = df2.rename(columns={'Age (years)': 'Age', 'Sex (m/f)': 'Sex'})

        # Select columns of interest and concatenate the two dataframes
        df1 = df1[['Subject ID','Age', 'Sex']]
        df2 = df2[['Subject ID','Age', 'Sex']]
        df = pd.concat([df1, df2], ignore_index=True)

        output_metadata_path = datasets_path / 'bap' / 'bap-metadata.csv'
        df.to_csv(output_metadata_path, index=False)

    elif dataset_name == 'lemon':
        pass # TODO


def main(args):
    for dataset_name in args.dataset_names:
        organize_metadata(args.datasets_path, dataset_name)


if __name__ == "__main__":
    # Creating an argument parser object
    parser = argparse.ArgumentParser(description="Metadata filtering script")
    dataset_names = ['bap', 'hbn', 'lemon']

    # Adding command line arguments
    parser.add_argument("--datasets_path", type=str, default='/data0/practical-sose23/brain-age/data/', 
                        help="Path to the datasets directory")
    
    parser.add_argument("--dataset_names", type=str, default=dataset_names, nargs='+', choices=dataset_names, 
                        help="List of dataset names")

    args = parser.parse_args()
    main(args)
