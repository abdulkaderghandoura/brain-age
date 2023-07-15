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

    # example urls
    urls = [
        "https://fcp-indi.s3.amazonaws.com/data/Archives/HBN/EEG/NDARAA396TWZ.tar.gz",
        "http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R10_Pheno.csv",
        "http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R1_1_Pheno.csv"
    ]

    # URL to the raw EEG files and phenotype files (containing age and subject id)
    url_hbn = 'https://fcp-indi.s3.amazonaws.com/data/Archives/HBN/EEG'
    url_pheno = 'http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno'

    releases = ["R1_1", "R2_1", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
    good_subjects = []

    # Loop over the listed dataset releases
    for release in releases:
        f_pheno = f"HBN_{release}_Pheno.csv"
        urllib.request.urlretrieve(f"{url_pheno}/{f_pheno}", f"{dest_path}/{f_pheno}")
        pheno_df = pd.read_csv(f"{dest_path}/{f_pheno}")
        subjects = sorted(pheno_df.EID)

        # Loop over subjects
        for sub in subjects:
            sub_url = f"{sub}.tar.gz"
            url = f"{url_hbn}/{sub_url}"
            out_path = dest_path / sub / "RSEEG"

            if not out_path.exists():
                os.makedirs(out_path)

            out_name = out_path / f"{sub}.tar.gz"

            if not out_name.is_file():
                try:
                    urllib.request.urlretrieve(url, out_name)
                    good_subjects.append(sub)
                    print(out_name)
                except Exception as err:
                    print(err)
                    print(f"release: {release}\nfile: {out_name}")

    # Save a log of subject IDs whose EEG data was successfully downloaded
    good_subs_df = pd.DataFrame(dict(subject=list(set(good_subjects))))
    good_subs_df.to_csv(dest_path / 'good_subjects.csv')

if __name__ == "__main__":
    main()
