from mne_realtime.externals import FieldTrip
import time
import numpy as np
from statistics import mean
import mne
from mne.io import RawArray, read_raw_bdf, read_raw_eeglab
from mne.connectivity import spectral_connectivity
from mne.time_frequency import psd_welch, psd_multitaper
import matplotlib.pyplot as plt

raw1 = read_raw_eeglab('/Users/jessicaye/Desktop/Datasets_set_examples/FASTER_170719_TRIAL2_trim_EM.set', preload=True)
raw2 = read_raw_eeglab('/Users/jessicaye/Desktop/Datasets_set_examples/FASTER_170719_TRIAL2_trim_MD.set', preload=True)

events_from_annot2, event_dict2 = mne.events_from_annotations(raw2)
info2 = mne.create_info(['STI'], raw2.info['sfreq'], ['stim'])
stim_data2 = np.zeros((1, len(raw2.times)))
stim_raw2 = RawArray(stim_data2, info2)
raw2.add_channels([stim_raw2], force_update_info=True)

mne.concatenate_raws([raw1, raw2])
# raw1.copy().pick_types(meg=False, stim=False).plot(start=3, duration=6)
# events1 = mne.find_events(raw1)
# print(events1[:5])  # show the first 5