# adapted from https://github.com/meeg-ml-benchmarks/brain-age-benchmark-paper/blob/main/download_data_lemon.py
import os
import pathlib
import urllib.request
import pandas as pd

# example urls
"https://fcp-indi.s3.amazonaws.com/data/Archives/HBN/EEG/NDARAA396TWZ.tar.gz"
"http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R10_Pheno.csv"
"http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R1_1_Pheno.csv"

url_pheno = "http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno"
url_hbn = "https://fcp-indi.s3.amazonaws.com/data/Archives/HBN/EEG"
data_path = pathlib.Path("/data0/practical-sose23/brain-age/data/hbn/HBN_RAW")

# if not data_path.exists():
#     os.makedirs(data_path)

releases = ["R1_1", "R2_1", "R3", "R4", "R5", "R6", "R7", "R8", "R9" "R10"]
good_subjects = list()
for release in releases:
    f_pheno = f"HBN_{release}_Pheno.csv"
    urllib.request.urlretrieve(f"{url_pheno}/{f_pheno}", 
                               f"{data_path}/{f_pheno}")
    pheno_df = pd.read_csv(f"{data_path}/{f_pheno}")
    subjects = sorted(pheno_df.EID)
    for sub in subjects:
        sub_url = f"{sub}.tar.gz"
        url = f"{url_hbn}/{sub_url}"
        out_path = data_path / sub / "RSEEG"
        if not out_path.exists():
            os.makedirs(out_path)
        out_name = out_path / f"{sub}.tar.gz"
        if not out_name.is_file():
            try:
                urllib.request.urlretrieve(url, out_name)
                good_subjects.append(sub)
            except Exception as err:
                print(err)
                print(f"release: {release}\nfile: {out_name}")

good_subs_df = pd.DataFrame(dict(subject=list(set(good_subjects))))
good_subs_df.to_csv('good_subjects.csv')