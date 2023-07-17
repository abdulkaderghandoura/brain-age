import argparse
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")

        end_time = time.time()
        execution_time = end_time - self.start_time
        print(f"Execution time: {execution_time} seconds")


def get_subject_id(file_path):
    file_path = Path(file_path).resolve()
    file_name = file_path.name
    dataset_name = file_path.parts[file_path.parts.index('data') + 1]

    if dataset_name == 'hbn':
        return file_path.parts[file_path.parts.index('rest') + 1]
    elif dataset_name == 'bap':
        if (file_path.parent.parent / f'{file_path.parent.name}.pickle').is_file():
            return file_name.split('_')[-4].lstrip('0')
        else:
            return file_name.split('_')[-3].lstrip('0')
    elif dataset_name == 'lemon':
        return file_name.split('_')[0].split('-')[1]


def add_labels_to_splits(args):
    datasets_path = Path(args.datasets_path).resolve()
    for dataset_name in tqdm(args.dataset_names):
        dataset_path = datasets_path / dataset_name
        metadata_input_path = dataset_path / f'{dataset_name}-metadata.csv'
        input_df = pd.read_csv(metadata_input_path, dtype={'Subject ID': str})

        # Select columns of interest
        input_df = input_df[['Subject ID','Age']]

        # Set 'Subject ID' as the index
        input_df = input_df.set_index('Subject ID')

        for split in ['train', 'val', 'test']:
            data_pairs = list()
            split_path = dataset_path / 'preprocessed' / args.d_version / f'{dataset_name}_{split}.txt'
            
            with open(split_path, 'r') as in_file:
                lines = in_file.readlines()

                for line in lines:
                    file_path = line.strip()
                    subject_id = get_subject_id(file_path)
                    assert subject_id in input_df.index.tolist()
                    subject_age = input_df.loc[subject_id, 'Age']
                    data_pairs.append((file_path, subject_age))
            
                output_df = pd.DataFrame(data_pairs, columns=['Subject ID','Age'])
                metadata_output_path = dataset_path / 'preprocessed' / args.d_version / f'{dataset_name}_{split}.csv'
                output_df.to_csv(metadata_output_path, index=False)

def get_midpoint(age_range):
    # Extract age from a range
    low, high = age_range.split("-")
    return (int(high)+int(low))/2 