def get_subject_id(file_path):
    dir_path, file_name = os.path.split(file_path)

    # Split current directory path into individual directories
    dirs = dir_path.split('/' if '/' in dir_path else '\\')

    dataset_name = dirs[dirs.index('data') + 1]

    if dataset_name == 'hbn':
        return dirs[dirs.index('rest') + 1]
    elif dataset_name == 'bap':
        return file_name.split('.')[-3]
