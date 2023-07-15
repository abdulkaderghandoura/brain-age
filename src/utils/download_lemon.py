import argparse
import os
import pathlib
import urllib.request
import pandas as pd

def main():
    # parse the destination directory
    parser = argparse.ArgumentParser(description='Download Script Parser')
    parser.add_argument('--dest', type=str, help='Destination directory')
    args = parser.parse_args()
    dest_path = pathlib.Path(args.dest)

    # URL to the raw EEG files
    url_eeg = 'https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/'

    # Provide a mapping between initial and current subject IDs
    url_name_map = 'https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/name_match.csv'
    urllib.request.urlretrieve(url_name_map, dest_path / "name_match.csv")
    name_match = pd.read_csv(dest_path / "name_match.csv")

    subjects = sorted(name_match.Initial_ID)
    extensions = ["eeg", "vhdr", "vmrk"]
    good_subjects = []

    # Loop over subjects
    for sub in subjects:
        for ext in extensions:
            sub_url = f"{sub}/RSEEG/{sub}.{ext}"
            url = f"{url_eeg}{sub_url}"
            out_path = dest_path / sub / "RSEEG"

            if not out_path.exists():
                os.makedirs(out_path)

            out_name = out_path / f"{sub}.{ext}"

            try:
                urllib.request.urlretrieve(url, out_name)
                good_subjects.append(sub)
                print(out_name)
            except Exception as err:
                print(err)

    # Save a log of subject IDs whose EEG data was successfully downloaded
    good_subj_df = pd.DataFrame(dict(subject=list(set(good_subjects))))
    good_subj_df.to_csv(dest_path / 'good_subjects.csv')

if __name__ == "__main__":
    main()
