import tarfile
import multiprocessing

def extract_tar_gz(file_path, extract_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extract_path)

# Usage example
file_path = '/data0/practical-sose23/brain-age/data/hbn/raw/NDARZT940RZG/RSEEG/NDARZT940RZG.tar.gz'
# extract_path = '/u/home/ghan/brain-age/temp'
extract_path = '/data0/practical-sose23/brain-age/data/hbn/'

# Set the number of worker processes (adjust according to your system)
num_processes = multiprocessing.cpu_count()

# Create a process pool
pool = multiprocessing.Pool(processes=num_processes)

# Extract files in parallel
pool.apply_async(extract_tar_gz, (file_path, extract_path))

# Close the pool and wait for all processes to complete
pool.close()
pool.join()
