import numpy as np
import mne
import scipy
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels

class InterpolateElectrodes:
    """
    interpolates between electrodes by recomputing the interpolation matrix for each sample
    """
    
    def __init__(self, from_montage, to_montage, to_channel_ordering, chs_to_exclude):

        ### Get interpolation matrix given several mne montage (covering all channels of interest)
        self.chs_to_exclude = chs_to_exclude
            
        self.from_ch_pos = np.stack(
            [value for key, value in from_montage.get_positions()["ch_pos"].items() \
             if not key in self.chs_to_exclude]
        )
        
        ch_name_to_pos = to_montage.get_positions()["ch_pos"]
        self.to_ch_pos = np.stack(
            [ch_name_to_pos[ch_name] for ch_name in to_channel_ordering]
        )
        
        self.interpolation_matrix = mne.channels.interpolation._make_interpolation_matrix(
                self.from_ch_pos, self.to_ch_pos
                )
    def __call__(self, x):
        x_interpolated = np.matmul(self.interpolation_matrix, x)
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
        chs_to_exclude = chs_to_exclude + [ch for ch in self.montage_hbn.ch_names if "Fid" in ch]
        # set-up interpolation
        self.interpolate = InterpolateElectrodes(montage_hbn, montage_bap, channel_names_bap, chs_to_exclude)
    
    def read_eeg(self, file):
        raw = mne.io.read_raw(file, verbose=False, preload=True)
        raw.set_montage(self.montage_hbn)
        return raw
    
    def read_mat(self, file):
        # Load the .mat file
        mat_struct = scipy.io.loadmat(file)

        ## organize the meta data
        raw_data = mat_struct["EEG"]["data"][0][0]
        fs = mat_struct["EEG"]["srate"]
        
        ## initialize raw hbn
        ch_types_hbn = len(self.ch_names_hbn)* ["eeg"]
        info = mne.create_info(ch_names=self.ch_names_hbn, sfreq=fs, ch_types=ch_types_hbn)
        info.set_montage(self.montage_hbn)
        return mne.io.RawArray(raw_data, info)
        

    def load_to_raw(self, file):
        ## Load the eeg file
        if file.suffix == ".mat":
            return self.read_mat(file)
        else:
            return self.read_eeg(file)
        
        
    def __call__(self, file):
        ## Load the eeg file
        raw = self.load_to_raw(file)

        ## find & repair bad channels (takes most of the processing time)
        nc = NoisyChannels(raw)
        nc.find_all_bads()
        raw.info['bads'].extend(nc.get_bads())
        raw.interpolate_bads(reset_bads=True)
        raw.compute_psd().plot_topomap(dB=True)
        plt.show()

        
        if self.interpolation:
            ch_types_bap = len(self.channel_names_bap)*["eeg"]
            info = mne.create_info(
                ch_names=self.channel_names_bap, 
                sfreq=raw.info["sfreq"], 
                ch_types=ch_types_bap
                )
            info.set_montage(self.montage_bap)
            raw_data = self.interpolate(raw._data)
            ## overwrite as raw bap
            raw = mne.io.RawArray(raw_data, info)
        
        return raw
