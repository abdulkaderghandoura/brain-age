import numpy as np
import mne
import scipy
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels

class InterpolateElectrodes:
    """
    interpolates between electrodes by recomputing the interpolation matrix for each sample
    """
    
    def __init__(self, from_montage, to_montage, chs_to_exclude):

        ### Get interpolation matrix given several mne montage (covering all channels of interest)
        self.chs_to_exclude = chs_to_exclude
        self.from_ch_pos = np.stack(
            [value for key, value in from_montage.get_positions()["ch_pos"].items() \
             if not key in self.chs_to_exclude]
        )
        self.to_ch_pos = np.stack(
            [value for key, value in to_montage.get_positions()["ch_pos"].items() \
             if not key in self.chs_to_exclude]
        )
        self.interpolation_matrix = mne.channels.interpolation._make_interpolation_matrix(
                self.from_ch_pos, self.to_ch_pos
                )
    def __call__(self, x):
        x_interpolated = np.matmul(self.interpolation_matrix, x)
#         plt.title("hbn data with repaired channels")
#         plt.imshow(x, aspect="auto")
#         plt.xlabel("time [@ 512Hz]")
#         plt.ylabel("# channels")
        plt.show()
        return x_interpolated

class HBNFormatter:
    
    def __init__(self, channel_names_bap, montage_bap, montage_hbn, interpolation=True):
        
        self.montage_bap = montage_bap 
        self.montage_hbn = montage_hbn
        self.channel_names_bap = channel_names_bap
        self.ch_names_hbn = [ch for ch in self.montage_hbn.ch_names if "Fid" not in ch]
        self.interpolation = interpolation
        
        # exclude duplicate locations and fiducials from interpolation
        chs_to_exclude = list(set(montage_bap.ch_names).difference(set(channel_names_bap)))
        chs_to_exclude = chs_to_exclude + ['FidNz', 'FidT9', 'FidT10']
        # set-up interpolation
        self.interpolate = InterpolateElectrodes(montage_hbn, montage_bap, chs_to_exclude)
        
        
        
    def __call__(self, mat_file):
        ## Load the .mat file
        mat_struct = scipy.io.loadmat(mat_file)

        ## organize the meta data
        raw_data = mat_struct["EEG"]["data"][0][0]
        fs = mat_struct["EEG"]["srate"]
        
        ## initialize raw hbn
        ch_types_hbn = len(self.ch_names_hbn)* ["eeg"]
        info = mne.create_info(ch_names=self.ch_names_hbn, sfreq=fs, ch_types=ch_types_hbn)
        info.set_montage(self.montage_hbn)
        raw = mne.io.RawArray(raw_data, info)
        
        ## find & repair bad channels (takes most of the processing time)
#         plt.title("raw hbn data")
#         plt.imshow(raw._data, aspect="auto")
#         plt.xlabel("time [@ 512Hz]")
#         plt.ylabel("# channels")
#         plt.show()
        nc = NoisyChannels(raw)
        nc.find_all_bads()
        raw.info['bads'].extend(nc.get_bads())
        raw.interpolate_bads(reset_bads=True)

        if self.interpolation:
            ch_types_bap = len(self.channel_names_bap)*["eeg"]
            info = mne.create_info(ch_names=self.channel_names_bap, sfreq=fs, ch_types=ch_types_bap)
            info.set_montage(self.montage_bap)
            raw_data = self.interpolate(raw._data)
            ## overwrite as raw bap
            raw = mne.io.RawArray(raw_data, info)
        
#         plt.title("interpolated hbn data with repaired channels")
#         plt.imshow(raw._data, aspect="auto")
#         plt.xlabel("time [@ 512Hz]")
#         plt.ylabel("# channels")
#         plt.show()
        
        return raw
