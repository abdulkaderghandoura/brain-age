# Import necessary modules
import os
import mne
import pickle
from tqdm import tqdm
import argparse
import json
import time
import warnings

from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_file(file_path):
    raw = mne.io.read_raw_brainvision(file_path, preload=True)

def preprocess_data(input_data_path, output_data_path, filters):
    start_time = time.time()
    # Disable RuntimeWarning messages
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Split input and output paths into directories
    input_dirs = input_data_path.split('/' if '/' in input_data_path else '\\')
    # output_dirs = output_data_path.split('/' if '/' in output_data_path else '\\')
    # Create a thread pool executor
    with ThreadPoolExecutor() as executor: # // max_workers=10
        # List to store the futures of the parallel tasks
        futures = Queue()
        # Walk through the directory tree starting from input_data_path
        for dir_path, _, file_names in os.walk(input_data_path):
            # Split current directory path into individual directories
            dirs = dir_path.split('/' if '/' in dir_path else '\\')
            if 'preprocessed' in dirs:
                for file_name in tqdm(file_names):
                    # Check if the file extension is '.vhdr'
                    if os.path.splitext(file_name)[1] == ".vhdr":
                        # Get the full path of the current file
                        file_path = os.path.join(dir_path, file_name)

                        future = executor.submit(preprocess_file, file_path)
                        futures.put(future)
        
        # Wait for all tasks to complete
        # for future in tqdm(as_completed(futures)):
        while not futures.empty():
            future = futures.get()
            # Retrieve any exceptions that occurred during processing
            exception = future.exception()
            if exception:
                print(f"An error occurred: {exception}")

        # Shutdown the executor
        executor.shutdown()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


def preprocess_from_command_line():
    # Creating an argument parser object
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    # Adding command line arguments
    parser.add_argument("input_data_path", help="Path to the input data directory")
    parser.add_argument("output_data_path", help="Path to the output data directory")
    parser.add_argument("--filters", default='{}', help="Optional key-value pairs in JSON format")
    # Parsing the command line arguments
    args = parser.parse_args()
    # Loading the JSON-formatted filters
    filters = json.loads(args.filters)
    # Preprocessing the data using the provided arguments
    preprocess_data(args.input_data_path, args.output_data_path, filters)


if __name__ == "__main__":
    preprocess_from_command_line()