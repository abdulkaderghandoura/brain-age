import os
import pandas as pd
from tqdm import tqdm

def get_dirs(dir_path):
    # Split datasets_path into directories
    dirs = dir_path.split('/' if '/' in dir_path else '\\')
    dirs = [item for item in dirs if item != '']
    return dirs


def get_subject_id(file_path):
    dir_path, file_name = os.path.split(file_path)
    dirs = get_dirs(dir_path)

    dataset_name = dirs[dirs.index('data') + 1]

    if dataset_name == 'hbn':
        return dirs[dirs.index('rest') + 1]
    elif dataset_name == 'bap':
        if os.path.exists(os.path.join(os.path.sep, *dirs[:-1], dirs[-1] + '.npy')):
            return file_name.split('_')[-4].lstrip('0')
        else:
            return file_name.split('_')[-3].lstrip('0')


def add_labels_to_splits(datasets_path, dataset_name, d_version):
    dataset_path = os.path.join(*get_dirs(datasets_path), dataset_name)
    metadata_input_path = os.path.join(os.path.sep, dataset_path, f'{dataset_name}-metadata.csv')
    input_df = pd.read_csv(metadata_input_path)

    # Select columns of interest
    input_df = input_df[['Subject ID','Age']]

    # Set 'Subject ID' as the index
    input_df = input_df.set_index('Subject ID')

    for split in tqdm(['train', 'val', 'test']):
        data_pairs = list()
        split_path = os.path.join(os.path.sep, dataset_path, 'preprocessed', d_version, f'{dataset_name}_{split}.txt')
        
        with open(split_path, 'r') as in_file:
            lines = in_file.readlines()

            for line in lines:
                file_path = line.strip()
                subject_id = get_subject_id(file_path)
                assert subject_id in input_df.index.tolist()
                subject_age = input_df.loc[subject_id, 'Age']
                data_pairs.append((file_path, subject_age))
        
            output_df = pd.DataFrame(data_pairs, columns=['Subject ID','Age'])
            metadata_output_path = os.path.join(os.path.sep, dataset_path, 'preprocessed', d_version, f'{dataset_name}_{split}.csv')
            output_df.to_csv(metadata_output_path, index=False)


if __name__ == "__main__":
    add_labels_to_splits('/data0/practical-sose23/brain-age/data', 'bap', 'v1.0')
    add_labels_to_splits('/data0/practical-sose23/brain-age/data', 'hbn', 'v1.0')

    # with open('/data0/practical-sose23/brain-age/data/bap/preprocessed/v1.0/bap_train.txt', 'r') as in_file:
    #     lines = in_file.readlines()
    #     print(len(lines))

    # df00 = pd.read_csv('/data0/practical-sose23/brain-age/data/bap/preprocessed/v1.0/bap_train.csv')
    # print(len(df00))
