# Import necessary modules
import os
import mne
import pickle
from tqdm import tqdm
import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed


def preprocess_file(file_path, sfreq=None, references=[], filters={}, new_dir_path=None):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Load EEG data into an EEG raw object
    eeg_obj = mne.io.read_raw_brainvision(file_path, verbose=False, preload=True)

    # Set channel types for all channels to 'eeg'
    eeg_obj.set_channel_types({channel: 'eeg' for channel in eeg_obj.ch_names})
    # Correct channel types of LE and RE
    eeg_obj.set_channel_types({'LE':'misc', 'RE':'misc'})

    for ref_item in references:
        if ref_item == 'average':
            channel_names = eeg_obj.ch_names
            eeg_obj.set_eeg_reference(ref_channels="average")
            eeg_obj = eeg_obj.pick(picks=channel_names)
        elif ref_item == 'median':
            # median_channel = np.median(eeg_obj.get_data(), axis=0)
            # eeg_obj -= median_channel
            pass # TODO
        elif ref_item in channel_names:
            channel_names = eeg_obj.ch_names
            eeg_obj.set_eeg_reference(ref_channels=ref_item)
            eeg_obj = eeg_obj.pick(picks=channel_names)

    # Apply the specified filters
    # Note: The if-else is implemented this way to minimize the number filtering operations
    if 'nf' in filters:
        # Apply a notch filter to the raw data
        eeg_obj = eeg_obj.notch_filter(freqs=filters['nf'], notch_widths=0.5, method='spectrum_fit')
    if 'lpf' in filters and 'hpf' in filters:
        # Apply a band-pass filter to the raw data
        eeg_obj = eeg_obj.filter(l_freq=filters['hpf'], h_freq=filters['lpf'], fir_design='firwin')
    elif 'lpf' in filters:
        # Apply a low-pass filter to the raw data
        eeg_obj = eeg_obj.filter(l_freq=None, h_freq=filters['lpf'], fir_design='firwin')
    elif 'hpf' in filters:
        # Apply a high-pass filter to the raw data
        eeg_obj = eeg_obj.filter(l_freq=filters['hpf'], h_freq=None, fir_design='firwin')

    if sfreq is not None:
        eeg_obj = eeg_obj.resample(sfreq=sfreq)

    new_file_path = None
    if new_dir_path is not None:
        # Create the new file path for the output file with a '.pickle' extension
        new_file_path = os.path.join(new_dir_path, os.path.splitext(file_name)[0] + '.pickle')

    return eeg_obj, new_file_path


def preprocess_data(input_data_path, output_data_path, sfreq, references, filters):
    start_time = time.time()
    # Disable RuntimeWarning messages
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Split input and output paths into directories
    input_dirs = input_data_path.split('/' if '/' in input_data_path else '\\')
    output_dirs = output_data_path.split('/' if '/' in output_data_path else '\\')
    
    # Create a thread pool executor
    with ThreadPoolExecutor() as executor: # // max_workers=10
        # List to store the futures of the parallel tasks
        futures = []
        # Walk through the directory tree starting from input_data_path
        for dir_path, _, file_names in os.walk(input_data_path):
            # Split current directory path into individual directories
            dirs = dir_path.split('/' if '/' in dir_path else '\\')
            # Process only data in preprocessed directories in our dataset
            if 'preprocessed' in dirs:
                for file_name in tqdm(file_names):
                    # Check if the file extension is '.vhdr'
                    if os.path.splitext(file_name)[1] == ".vhdr":
                        # Get the full path of the current file
                        file_path = os.path.join(dir_path, file_name)
                        # Create the new directory path for the output file
                        new_dir_path = os.path.join(*output_dirs, *dirs[len(input_dirs):])
                        new_dir_path = os.path.abspath(new_dir_path)
                        os.makedirs(new_dir_path, exist_ok=True)
                        future = executor.submit(preprocess_file, file_path, sfreq, references, filters, new_dir_path)
                        futures.append(future)

        # Wait for all tasks to complete
        for future in as_completed(futures):
            # Retrieve the returned value from the future
            eeg_obj, new_file_path = future.result()
            # # Save the filtered EEG data as a pickle file
            # with open(new_file_path, mode='wb') as out_file:
            #     pickle.dump(eeg_obj, out_file, protocol=pickle.DEFAULT_PROTOCOL)
            # Retrieve any exceptions that occurred during processing
            exception = future.exception()
            if exception:
                print(f"An error occurred: {exception}")
        
    # Shutdown the executor
    executor.shutdown()

    # Calculate the elapsed time in seconds
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


def preprocess_from_command_line():
    # Creating an argument parser object
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    # Adding command line arguments
    parser.add_argument("input_data_path", help="Path to the input data directory")
    parser.add_argument("output_data_path", help="Path to the output data directory")
    parser.add_argument("--downsampling", help="Factor for downsampling")
    parser.add_argument("--references", nargs='+', default=[], help="List of reference channels")
    parser.add_argument("--filters", default='{}', help="Optional key-value pairs in JSON format")
    # Parsing the command line arguments
    args = parser.parse_args()
    # Loading the JSON-formatted filters
    filters = json.loads(args.filters)
    # Preprocessing the data using the provided arguments
    preprocess_data(args.input_data_path, args.output_data_path, args.downsampling, args.references, filters)


if __name__ == "__main__":
    preprocess_from_command_line()