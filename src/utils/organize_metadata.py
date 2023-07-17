import pandas as pd
from tqdm import tqdm
# from utils import get_dirs
from utils import get_midpoint
from pathlib import Path
import argparse

def organize_metadata(datasets_path, dataset_name):
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
        input_metadata_path = raw_dir_path / 'clinical_data_updated_2020-08-04.ods'
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
        # Read data from Excel sheets
        input_metadata_dir = datasets_path / 'lemon' / 'Behavioural_Data_MPILMBB_LEMON'
        df = pd.read_csv(input_metadata_dir / 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
        df_names = pd.read_csv(input_metadata_dir.parent / 'name_match.csv')

        # Update subject names to match file names
        namemap = {old:new for old, new in zip(df_names['INDI_ID'], df_names['Initial_ID'])}
        df['Subject ID'] = df['Unnamed: 0'].apply(lambda x:namemap[x].split('-')[-1])
        # Extract the midpoint of the age range as the target
        df['Age'] = df['Age'].apply(get_midpoint)
        # Convert sex column to categories
        df['Sex'] = df['Gender_ 1=female_2=male'].apply(lambda x: 'f' if x==1 else 'm')
        # Extract only the columns of interest from the meta-data
        df = df[['Subject ID','Age', 'Sex']]

        output_metadata_path = datasets_path / 'lemon' / 'lemon-metadata.csv'
        df.to_csv(output_metadata_path, index=False)


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
    
    parser.add_argument("--dataset_names", type=str, nargs='+', choices=dataset_names, required=True, 
                        help="List of dataset names")

    args = parser.parse_args()
    main(args)
