# from https://github.com/meeg-ml-benchmarks/brain-age-benchmark-paper/blob/main/download_data_lemon.py
import os
import pathlib
import urllib.request
import pandas as pd


DEBUG = False
url_lemon = ('https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON'
             '/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/')

lemon_info = pd.read_csv(
  "/data0/practical-sose23/brain-age/data/lemon/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")

name_match = pd.read_csv(
  "/data0/practical-sose23/brain-age/data/lemon/name_match.csv")

data_path = pathlib.Path("/data0/practical-sose23/brain-age/data/lemon/LEMON_RAW")

# if not data_path.exists():
#     os.makedirs(data_path)

# lemon_info.rename({"Unnamed: 0":"ID"}, axis=1, inplace=True)
subjects = sorted(name_match.Initial_ID)
if DEBUG:
    subjects = subjects[:1]

extensions = ["eeg", "vhdr", "vmrk"]
good_subjects = list()

for sub in subjects:
    for ext in extensions:
        sub_url = f"{sub}/RSEEG/{sub}.{ext}"
        url = f"{url_lemon}{sub_url}"
        out_path = data_path / sub / "RSEEG"
        if not out_path.exists():
            os.makedirs(out_path)
        out_name = out_path / f"{sub}.{ext}"
        print(url)
        try:
            urllib.request.urlretrieve(url, out_name)
            good_subjects.append(sub)
        except Exception as err:
            print(err)

good_subs_df = pd.DataFrame(dict(subject=list(set(good_subjects))))
good_subs_df.to_csv('good_subjects.csv')